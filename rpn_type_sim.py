# rpn_type_sim.py
import inspect
import functools
from typing import Tuple, Sequence, Dict
import operators
from tokenizer import AlphaTokenizer

class RPNTypeSimulator:
    """
    轻量级 RPN 类型推演器，用于 token-by-token 合法性检查。
    类型符号:
      S = Series (行情序列)
      C = Constant (数值常量)
      B = Bool (布尔序列，扩展用)
    """
    S = "S"
    C = "C"
    B = "B"

    def __init__(self, tokenizer: AlphaTokenizer, const_param_names='window'):
        # tokenizer 用于识别 base_fields、special_tokens
        self.tok = tokenizer
        self.base_fields = set(tokenizer.base_fields)
        self.const_prefix = "CONST_"
        self.special_tokens = set(tokenizer.special_tokens)
        # 构建算子签名表
        self.sig_table = self._build_signature_table(const_param_names)

    def _build_signature_table(self, const_param_names) -> Dict[str, Tuple[Tuple[str,...], str]]:
        table: Dict[str, Tuple[Tuple[str,...], str]] = {}
        # 从 operators.py 自动读取函数签名
        for name, fn in inspect.getmembers(operators, inspect.isfunction):
            if name.startswith("_"):
                continue
            params = list(inspect.signature(fn).parameters.keys())
            if not params:
                continue
            # 参数第一个默认 Series，其余根据名称判定
            in_types = []
            for p in params:
                if p == const_param_names:
                    in_types.append(self.C)
                else:
                    in_types.append(self.S)
            table[name] = (tuple(in_types), self.S)
        # 四则运算覆盖
        for op in ('+', '-', '*', '/'):
            table[op] = ((self.S, self.S), self.S)
        return table

    @functools.lru_cache(maxsize=16384)
    def simulate(self, tokens: Tuple[str, ...]) -> Tuple[str, ...] | None:
        """
        对前缀 tokens 做类型推演，返回类型栈（tuple），非法返回 None。
        """
        stack: list[str] = []
        for tk in tokens:
            if tk in self.special_tokens:
                continue
            elif tk in self.base_fields:
                stack.append(self.S)
            elif tk.startswith(self.const_prefix) or self._is_float(tk):
                stack.append(self.C)
            elif tk in self.sig_table:
                in_types, out = self.sig_table[tk]
                if len(stack) < len(in_types):
                    return None
                # 类型匹配 (反向)
                for need, got in zip(reversed(in_types), reversed(stack[-len(in_types):])):
                    if need != got:
                        return None
                # 出栈并入栈
                del stack[-len(in_types):]
                stack.append(out)
            else:
                return None
        return tuple(stack)

    def is_valid_append(
        self, prefix: Sequence[str], new_token: str, remaining_steps: int
    ) -> bool:
        """
        判断在 prefix 后追加 new_token 是否仍
        
        
        有可能归约到单一 Series (['S'])。
        remaining_steps = 可放入的 payload token 数 (不含 [SEP])
        """
        pre = self.simulate(tuple(prefix))
        if pre is None:
            return False
        post = self.simulate(tuple(prefix + [new_token]))
        if post is None:
            return False
        depth = len(post)
        # 最坏情况：每步栈深 -1，再 +1 步 [SEP]
        need = (depth - 1) + 1
        return need <= remaining_steps

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
