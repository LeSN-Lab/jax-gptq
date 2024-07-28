from collections import defaultdict
from copy import deepcopy
from functools import partial, reduce
import numpy as np
import warnings
import re
import jax
import jax.numpy as jnp
from jax._src.core import Literal
from jax.util import safe_map
from tqdm import tqdm
import re

from .gptq import gptq, pack_matrix, QuantizedMatrix

def tree_size_bytes(tree):
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(
            lambda x: x.size * x.itemsize,
            tree
        ),
        0
    )

def quantize(
    fn,
    params,
    inputs,
    block_size=128,
    actorder=False,
    damping=0.01,
    use_quantized_activations=True,
    use_fp64=False,
    use_params_fp32=False,
    exclude_layers=None,
):
    """
    Run the GPT-Q algorithm on a function to produce quantized versions of its parameters
    Arguments:
        fn: The function to be transformed. It should take two arguments:
            1. A pytree of parameters to be quantized. This corresponds to the `params` pytree from libraries such as Flax/Haiku
            2. A pytree of other arguments. If the original model takes more than one extra argument, you can write a wrapper which takes a tuple as the second argument. TODO: handle varargs
        params: The params pytree. Buffers in this tree may be freed to save memory, so do not re-use it after calling this function.
        inputs: A list of batches of inputs. If your model needs to be vmapped to handle batches, do that before calling quantize.
    """

    with jax.disable_jit():
        jaxpr_args = (params, inputs[0])
        if use_params_fp32:
            jaxpr_args = jax.tree_util.tree_map(
                lambda x: jax.ShapeDtypeStruct(x.shape, jnp.float32) if x.dtype.kind == 'f' else x,
                jaxpr_args
            )
        closed_jaxpr = jax.make_jaxpr(fn)(*jaxpr_args)
    params = jax.device_put(params, jax.devices('cpu')[0])
    inputs = jax.device_put(inputs, jax.devices('cpu')[0])

    argnums = set()
    param_args, param_struct = jax.tree_util.tree_flatten(params)
    input_args = [jax.tree_util.tree_leaves(inp) for inp in inputs]
    input_args = [list(arg) for arg in zip(*input_args)]

    argnums = set(range(0, len(param_args)))

    result = _eval_and_quantize(
        closed_jaxpr.jaxpr,
        closed_jaxpr.literals,
        argnums,
        *param_args,
        *input_args,
        block_size=block_size,
        actorder=actorder,
        damping=damping,
        use_quantized_activations=use_quantized_activations,
        use_fp64=use_fp64,
        use_params_fp32=use_params_fp32,
        exclude_layers=exclude_layers
    )
    for ind, quantized_param in result.items():
        param_args[ind] = quantized_param

    return jax.tree_util.tree_unflatten(param_struct, param_args)

def _get_delete_points(jaxpr):
    deps = defaultdict(set)
    for i, eqn in enumerate(jaxpr.eqns):
        for var in set(v for v in eqn.invars if not isinstance(v, Literal)):
            deps[var].add(i)

    deps = dict(deps)
    delete_vars = []
    for i, eqn in enumerate(jaxpr.eqns):
        eqn_delete = []
        for var in set(v for v in eqn.invars if not isinstance(v, Literal)):
            deps[var].remove(i)
            if not deps[var]:
                eqn_delete.append(var)
                del deps[var]
        delete_vars.append(eqn_delete)
    return delete_vars

def _maybe_delete(val):
    if not val.is_deleted():
        val.addressable_data(0).delete()

def _eval_and_quantize(
    jaxpr,
    consts,
    argnums,
    *args,
    block_size=128,
    actorder=False,
    damping=0.01,
    use_quantized_activations=True,
    use_fp64=False,
    use_params_fp32=False,
    exclude_layers=None  # 새로운 인자: 양자화에서 제외할 레이어 목록
):
    # CPU와 GPU 디바이스 설정
    cpu = jax.devices('cpu')[0]
    gpu = jax.devices('gpu')[0]
    # Args are all either params or lists of tensors
    # 결과를 저장할 딕셔너리 초기화
    quantized_results = {}
    name_to_pos = {}
    
    # 배치 수 계산
    n_batches = len(next(a for i, a in enumerate(args) if i not in argnums))

    # GPU에서 사용할 환경 초기화
     # Everything in here should be on GPU
    envs = [{} for _ in range(n_batches)]

    # Map from var name to a tuple of value, original_name, and a stack of transformations to map it back to orig param shape
    # 파라미터 환경 초기화: 변수 이름에 대한 (값, 원래 이름, 변환 스택) 매핑
    param_env = {}

    # 입력 인자 처리
    for index, name in enumerate(jaxpr.invars):
        if index in argnums:
            param_env[name] = (args[index], name, ())
            name_to_pos[name] = index
        else:
            for i in range(n_batches):
                envs[i][name] = args[index][i]
    # 메모리 해제 함수 정의
    def delete(name):
        if name not in envs[0]:
            return
        for env in envs:
            env[name].addressable_data(0).delete()
            del env[name]

    # 삭제 지점 계산
    delete_points = _get_delete_points(jaxpr)

    # 상수 환경 설정
    const_env = {name: val for name, val in zip(jaxpr.constvars, consts)}
    
    # 메인 루프 시작
    pos = 0
    bar = tqdm(desc='Quantizing')
    while True:
        bar.update(1)
        # 다음 행렬 곱셈 연산 찾기
        next_pos, needed_names, matmul_handler, updated_param_env = update_params_to_next_matmul(
            eqns=jaxpr.eqns,
            start_pos=pos,
            delete_points=delete_points,
            param_env=param_env,
            env=envs[0],
            exclude_layers=exclude_layers
        )
        if next_pos is None:
            break
        
        # 필요한 파라미터를 GPU로 이동
        block_param_env = {
            name: jax.device_put(param_env[name][0], gpu)
            for name in needed_names if name in param_env
        }
        if use_params_fp32:
            for k, v in block_param_env.items():
                if v.dtype.kind == 'f':
                    block_param_env[k] = v.astype(jnp.float32)

        print(f'Current env size: {tree_size_bytes(envs):.2e} bytes')
        print(f'Current param env size: {tree_size_bytes(block_param_env):.2e} bytes')
        delete_keys = set(var for i in range(pos, next_pos) for var in delete_points[i])
        # 세그먼트 실행
        segment_eqns = jaxpr.eqns[pos:next_pos]

        # If a parameter has been transformed keep it in the param env instead of the individual envs
        drop_env_keys = set(k for k in updated_param_env if k not in param_env)
        missing_keys = set(k for k in param_env if k not in updated_param_env)

        block_fn = jax.jit(partial(run_segment, segment_eqns, pos, delete_points, drop_env_keys))
        for i, env in enumerate(envs):
            gpu_env = jax.device_put(env, gpu)
            new_env = block_fn(block_param_env, gpu_env, const_env)
            envs[i] = new_env
            #envs[i] = jax.device_put(new_env, cpu)
            #jax.tree_map(_maybe_delete, (gpu_env, new_env))

        # 메모리 정리
        for param in block_param_env.values():
            param.addressable_data(0).delete()

        del block_param_env

        param_env = updated_param_env

        #(jax.device_put(0., gpu) + 0).block_until_ready()
        # 행렬 곱셈 연산 처리
        matmul_eqn = jaxpr.eqns[next_pos]

        all_args = []
        if sum(argname in param_env for argname in matmul_eqn.invars) > 1:
            raise NotImplementedError('Currently only one quantize target is supported per op')
         
         # 양자화 대상 확인
        quantize_argname = next(argname for argname in matmul_eqn.invars if argname in param_env)


        # 인자 준비
        
        for argname in matmul_eqn.invars:
            if argname in param_env:
                all_args.append(param_env[argname][0])
            else:
                all_args.append([env[argname] for env in envs])
        all_args = [jax.device_put(arg, gpu) for arg in all_args]



        
        handler_coro = matmul_handler(all_args)

        w, xs = next(handler_coro)        
        #양자화 여부 결정
        # print(exclude_layers)
        should_quantize = True
        if exclude_layers:
            print("exclude_layers :", exclude_layers)
            for layer_pattern in exclude_layers:
                if re.match(layer_pattern, str(matmul_eqn), re.DOTALL):    
                    should_quantize = False
                    break

        # 양자화 수행 또는 건너뛰기
        if should_quantize:
            # print("해당 layer가 exclude_layer에 속하면 양자화하지 않습니다. : ", str(matmul_eqn))
            quantized_w, quantize_params = gptq(
                W=w,
                xs=xs,
                block_size=block_size,
                actorder=actorder,
                damping=damping,
                use_fp64=use_fp64
            )
        else:
            print("양자화안됨")
            quantized_w = w
            quantize_params = {'scale': jnp.ones_like(w), 'zero' : jnp.zeros_like(w)}
        assert quantized_w.shape == w.shape
        #결과 처리
        try:
            if matmul_eqn.primitive.name == 'conv_general_dilated':
                handler_coro.send((w, xs))
                # next(handler_coro)
                handler_coro.send((quantized_w, quantize_params['scale'], quantize_params['zero'], should_quantize))
            else:
                # next(handler_coro)
                # handler_coro.send((w, xs))
                handler_coro.send((quantized_w, quantize_params['scale'], quantize_params['zero']))
            assert False, 'Handler should have stopped'

        except StopIteration as e:
            quantized_w, quantize_params['scale'], quantize_params['zero'], contraction_axis = e.value

        #이 줄은 현재 연산의 출력 변수를 설정합니다.
        outvars = jaxpr.eqns[next_pos].outvars
        #양자화 대상이 아닌 입력 변수의 인덱스를 저장합니다.
        delete_indices = [i for i, name in enumerate(matmul_eqn.invars) if name != quantize_argname]



        do_eval = jax.jit(partial(eval_eqn, matmul_eqn))

        
        #양자화된 가중치를 준비하고 GPU로 이동시킵니다.
        matmul_w_arg = quantized_w if use_quantized_activations else param_env[quantize_argname][0]
        if use_params_fp32:
            matmul_w_arg = matmul_w_arg.astype(jnp.float32)

        matmul_w_arg = jax.device_put(matmul_w_arg, gpu)

        for env in envs:
            #GPU인자 준비
            gpu_args = [
                matmul_w_arg
                if argname == quantize_argname else
                env[argname]
                for argname in matmul_eqn.invars
            ]
            gpu_args = jax.device_put(gpu_args, gpu)
            
            #연산 수행
            results = do_eval(*gpu_args)

            #대용량 결과 처리
            if tree_size_bytes(results) > 1e8:
                # This should offload stuff like the final logits to the CPU
                #CPU로 오프로드
                cpu_results = jax.device_put(results, cpu)
                jax.tree_map(lambda x: x.is_deleted() or x.addressable_data(0).delete(), results)
                results = cpu_results
            #결과 저장
            if matmul_eqn.primitive.multiple_results:
                for outvar, value in zip(outvars, results):
                    env[outvar] = value
            else:
                env[outvars[0]] = results
            # 불필요한 데이터 삭제
            for name in delete_points[next_pos]:
                if name in env:
                    _maybe_delete(env[name])
                    del env[name]

            #for i in delete_indices:
            #    gpu_args[i].addressable_data(0).delete()
            #(jax.device_put(0., gpu) + 0).block_until_ready()

        #for name in delete_points[next_pos]:
        #    delete(name)

        # TODO: Instead of catching duplicate quantizations here avoid doing the calculation in the first place
        orig_w, orig_name, inv_transforms = param_env[quantize_argname]
        write_arg = name_to_pos[orig_name]
        if write_arg not in quantized_results:
            packed_result = pack_matrix(quantized_w, quantize_params, contraction_axis)
            un_transformed = reduce(lambda x, f: f(x), inv_transforms, packed_result)
            quantized_results[write_arg] = jax.device_put(un_transformed, cpu)

            if quantize_argname not in delete_points[next_pos]:
                cpu_quantized_w = jax.device_put(quantized_w, cpu)
                param_env[quantize_argname] = cpu_quantized_w, orig_name, inv_transforms
            orig_w.addressable_data(0).delete()
        elif quantize_argname in delete_points[next_pos]:
            orig_w.addressable_data(0).delete()
            del param_env[quantize_argname]

        quantized_w.addressable_data(0).delete()
        #(jax.device_put(0., gpu) + 0).block_until_ready()
        pos = next_pos + 1

    return quantized_results

def update_params_to_next_matmul(eqns, start_pos, delete_points, param_env, env, exclude_layers=None):
    # eqns: JAX의 방정식(연산) 목록
    # start_pos: 시작 위치
    # delete_points: 메모리 최적화를 위한 삭제 지점
    # param_env: 파라미터 환경
    # env: 실행 환경
    # exclude_layers: 양자화에서 제외할 레이어 패턴 목록
    """
    이 수정된 update_params_to_next_matmul 함수는 exclude_layers를 사용하여 특정 레이어를 양자화에서 제외할 수 있게 해줍니다.
    또한 should_quantize 플래그를 도입하여 각 연산에 대해 양자화 여부를 동적으로 결정하고, 이 정보를 핸들러에 전달합니다.
    이를 통해 모델의 특정 부분은 양자화하지 않고 원래의 정밀도를 유지할 수 있게 됩니다.
    """
    # 새로운 파라미터 환경 복사
    new_param_env = {k: v for k, v in param_env.items()}
    # 환경 내 변수들의 shape와 dtype 정보 추출
    env_shapes = {k: jax.ShapeDtypeStruct(v.shape, v.dtype) for k, v in env.items()}
    # 필요한 변수 이름들을 추적하기 위한 집합
    needed_names = set()
    
    # 주어진 방정식들을 순회
    for i, eqn in enumerate(eqns[start_pos:], start_pos):
        invars = eqn.invars
        op_name = eqn.primitive.name
        # 파라미터 변환 연산 처리

        if op_name in PARAM_TRANSFORMS:
            arg, = invars
            needed_names.add(arg)
            if arg in new_param_env and len(new_param_env[arg][0].shape) > 1:
                val, orig_name, transforms = new_param_env[arg]
                new_transform = PARAM_TRANSFORMS[op_name](eqn, val)
                new_name, = eqn.outvars
                new_val = eval_eqn(eqn, val)
                new_param_env[new_name] = new_val, orig_name, (transforms + (new_transform,))
                if arg in delete_points[i]: #TODO: Become certain that making this just a soft check was fine
                    del new_param_env[arg]
                else:
                    warnings.warn(f'Transformation `{op_name}` is applied to a target parameter of shape {new_param_env[arg][0].shape} which is later reused. This may lead to this parameter not being quantized, or it being quantized poorly.')
                continue
            
        # 인자 shape 정보 추출
        arg_shapes = [invar.aval for invar in invars]

        # 각 인자가 양자화 대상인지 확인
        args_are_targets = [(
            False if isinstance(v, Literal) else
            (v in new_param_env and len(new_param_env[v][0].shape) > 1)
        ) for v in invars]

        # 양자화 대상 연산 처리
        if any(args_are_targets):
            if op_name == 'pjit':
                warnings.warn(f'Quantization does not descend into pjit')
            if op_name in PRIMITIVE_TO_MATMUL:
                predicate, handler = PRIMITIVE_TO_MATMUL[op_name]
                if predicate(eqn, args_are_targets, arg_shapes):
                    # 양자화 여부 결정
                    should_quantize = True
                    if exclude_layers:
                        for layer_pattern in exclude_layers:
                            if re.match(layer_pattern, str(eqn), re.DOTALL):
                                should_quantize = False
                                break
                    # 결과 반환: 위치, 필요한 이름들, 핸들러 (should_quantize 포함), 새 파라미터 환경
                    return i, needed_names, partial(handler, eqn, args_are_targets, should_quantize), new_param_env
            else:
                warnings.warn(f'Operation {eqn.primitive.name} not supported for quantization')


        out_shapes = jax.eval_shape(partial(eval_eqn, eqn), *arg_shapes)
        if not eqn.primitive.multiple_results:
            out_shapes = [out_shapes]
        safe_map(env_shapes.__setitem__, eqn.outvars, out_shapes)
        needed_names.update(v for v in invars if not isinstance(v, Literal))
    return None, needed_names, None, None

def run_segment(eqns, start_pos, delete_points, drop_env_keys, param_env, env, const_env):
    env = dict(env)
    def read(v):
        if isinstance(v, Literal):
            return v.val
        if v in param_env:
            return param_env[v]
        if v in env:
            return env[v]
        return const_env[v]

    def write(v, val):
        env[v] = val

    for i, eqn in enumerate(eqns, start_pos):
        eqn_args = safe_map(read, eqn.invars)
        ans = eval_eqn(eqn, *eqn_args)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)

        for varname in delete_points[i]:
            if varname in env:
                del env[varname]
    for key in drop_env_keys:
        env.pop(key, None)
    return env

def dot_general_predicate(eqn, args_are_targets, args):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params['dimension_numbers']
    if sum(args_are_targets) > 1:
        warnings.warn('Quantizing two parameters which are multiplied together is not supported')
        return False
    if lhs_batch or rhs_batch:
        warnings.warn('Quantizing batched matmuls is not supported')
        return False
    if len(lhs_contract) > 1 or len(rhs_contract) > 1:
        warnings.warn('Quantizing dots with more than one contraction is not supported')
        return False

    return True


@partial(jax.jit, static_argnums=(1, 2))
def permute_to_matrix(w, permutation, keep_first):
    w = jnp.transpose(w, permutation)

    out_shape = (w.shape[0], -1) if keep_first else (-1, w.shape[-1])
    w = jnp.reshape(w, out_shape)
    return w

@partial(jax.jit, static_argnums=(1, 2))
def to_original_shape(w, shape, restore_permutation):
    return jnp.transpose(
        jnp.reshape(w, shape),
        restore_permutation
    )

def handle_dot_general(eqn, args_are_targets, should_quantize, args):
    lhs, rhs = args
    ((lhs_contract,), (rhs_contract,)), _ = eqn.params['dimension_numbers']
    if args_are_targets[0]:
        w, xs = lhs, rhs
        w_contract, x_contract = lhs_contract, rhs_contract
    else:
        w, xs = rhs, lhs
        w_contract, x_contract = rhs_contract, lhs_contract

    orig_w_shape = w.shape
    w_permutation = None
    if w_contract != 0 or len(w.shape) > 2:
        w_permutation = tuple([w_contract, *(i for i in range(len(w.shape)) if i != w_contract)])
        w = permute_to_matrix(w, w_permutation, True)

    assert isinstance(xs, list)

    x_permutation = None
    if x_contract != len(xs[0].shape) - 1:
        x_permutation = tuple([*(i for i in range(len(xs[0].shape)) if i != x_contract), x_contract])

    prepared_xs = []
    for x in xs:
        if x_permutation is not None:
            x = permute_to_matrix(x, x_permutation, False)
        prepared_xs.append(x)

    quantized_w, scales, zeros = yield w, prepared_xs
    
    if not should_quantize:
        quantized_w, scales, zeros = w, jnp.ones_like(w), jnp.zeros_like(w)
        
    if w_permutation:
        unpermute = tuple(np.argsort(w_permutation))
        shape = tuple(orig_w_shape[i] for i in w_permutation)
        quantized_w = to_original_shape(quantized_w, shape, unpermute)

        scale_shape = tuple(d for i, d in enumerate(orig_w_shape) if i != w_contract)
        scales = jnp.reshape(scales, scale_shape)
        zeros = jnp.reshape(zeros, scale_shape)

    return quantized_w, scales, zeros, int(w_contract)

def conv_predicate(eqn, args_are_targets, args):
    inp_is_target, kernel_is_target = args_are_targets
    if inp_is_target:
        warnings.warn('Only quantizing the kernel of a conv is supported, not the input')

    if not kernel_is_target:
        return False

    params = eqn.params
    if any(val != 1 for val in params['window_strides']):
        warnings.warn('Currently only quantizing convs with stride 1 is supported')
        return False

    if any(val != 1 for val in params['rhs_dilation']):
        warnings.warn('Currently only quantizing convs with dilation 1 is supported')
        return False

    if params['feature_group_count'] != 1:
        warnings.warn('Currently only quantizing convs with feature group count 1 is supported')
        return False

    if params['batch_group_count'] != 1:
        warnings.warn('Currently only quantizing convs with batch group count 1 is supported')
        return False

    # Each is: Batch, feature, spatial...
    kernel_spatial_dims = params['dimension_numbers'][1][2:]

    kernel_shape = args[1].shape
    for spatial_dim in kernel_spatial_dims:
        if kernel_shape[spatial_dim] != 1:
            warnings.warn('Currently only quantizing convs with 1x..x1 kernels are supported')
            return False

    return True

def handle_conv(eqn, args_are_targets, should_quantize, args):
    # eqn: 현재 처리 중인 convolution 연산
    # args_are_targets: 각 인자가 양자화 대상인지 여부
    # should_quantize: 이 특정 convolution을 양자화해야 하는지 여부
    # args: convolution 연산의 입력 인자들
    inps, kernel = args
    inp_shape = inps[0].shape
    kernel_shape = kernel.shape

    # convolution의 차원 정보 추출
    # (inp_batch_dim, inp_feature_dim, inp_spatial_dims), (kernel_out_dim, kernel_in_dim, *kernel_spatial_dims), _ = eqn.params['dimension_numbers']
    dim_numbers = eqn.params['dimension_numbers']
    inp_dims, kernel_dims, out_dims = dim_numbers[:3]
    inp_batch_dim, inp_feature_dim, *inp_spatial_dims = inp_dims
    kernel_out_dim, kernel_in_dim, *kernel_spatial_dims = kernel_dims

    # 커널을 2D 행렬로 변환
    flat_kernel = jnp.squeeze(kernel, kernel_spatial_dims)

    # 필요한 경우 커널 전치
    needs_transpose = kernel_out_dim < kernel_in_dim
    if needs_transpose:
        flat_kernel = flat_kernel.T

    # 입력 텐서 준비
    inp_permutation = None
    if inp_feature_dim != len(inp_shape) - 1:
        inp_permutation = tuple([*(i for i in range(len(inp_shape)) if i != inp_feature_dim), inp_feature_dim])

    prepared_inps = []
    for inp in inps:
        if inp_permutation is not None:
            inp = permute_to_matrix(inp, inp_permutation, False)
        prepared_inps.append(inp)
        
    # 첫 번째 yield: 평탄화된 커널과 준비된 입력 반환
    flat_kernel, prepared_inps = yield flat_kernel, prepared_inps

    # 양자화 수행 여부 결정
    if should_quantize:
        # 두 번째 yield: 양자화된 커널, 스케일, 제로 포인트 획득
        quantized_kernel, scales, zeros, should_quantize = yield
    else:
        # 양자화하지 않는 경우, 원본 값 사용
        quantized_kernel, scales, zeros = flat_kernel, jnp.ones_like(flat_kernel), jnp.zeros_like(flat_kernel)

    
    if needs_transpose:
        quantized_kernel = quantized_kernel.T
        scales = scales.T
        zeros = zeros.T
        
    print("현재 should_quantize 상태", should_quantize )
    
    if should_quantize:
        for dim in sorted(kernel_spatial_dims):
            quantized_kernel = jnp.expand_dims(quantized_kernel, dim)
            scale_dim = dim if dim < inp_feature_dim else dim - 1
            scales = jnp.expand_dims(scales, scale_dim)
            zeros = jnp.expand_dims(zeros, scale_dim)
    else:
        # 원본 커널 shape로 복원
        quantized_kernel = kernel
        scales = jnp.ones_like(kernel)
        zeros = jnp.zeros_like(kernel)


    return quantized_kernel, scales, zeros, kernel_in_dim

def eval_eqn(eqn, *args):
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    ans = eqn.primitive.bind(*subfuns, *args, **bind_params)
    return ans

PRIMITIVE_TO_MATMUL = {
    'dot_general': (dot_general_predicate, handle_dot_general),
    'conv_general_dilated': (conv_predicate, lambda eqn, args_are_targets, should_quantize, args: handle_conv(eqn, args_are_targets, should_quantize, args))
}

def inverse_transpose(eqn, arg):
    unpermute = tuple(np.argsort(eqn.params['permutation']))
    def inverse(quantized_matrix):
        prev_contract_axis = quantized_matrix.contraction_axis
        new_contraction_axis = unpermute[prev_contract_axis]
        new_int_weight = jax.lax.transpose(quantized_matrix.int_weight, permutation=unpermute)

        unpermute_scale = [
            i if i < prev_contract_axis else i - 1
            for i in unpermute
            if i != prev_contract_axis
        ]
        new_scale = jax.lax.transpose(quantized_matrix.scale, permutation=unpermute_scale)
        new_zero = jax.lax.transpose(quantized_matrix.zero, permutation=unpermute_scale)
        return QuantizedMatrix(
            int_weight=new_int_weight,
            scale=new_scale,
            zero=new_zero,
            contraction_axis=new_contraction_axis
        )

    return inverse

def inverse_convert_type(eqn, arg):
    return lambda x: x

PARAM_TRANSFORMS = {
    'transpose': inverse_transpose,
    'convert_element_type': inverse_convert_type,
}
