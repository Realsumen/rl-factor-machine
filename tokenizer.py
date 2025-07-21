# tokenizer.py
import re
import inspect
import operators
from typing import List, Dict

class AlphaTokenizer:
    """
    将 RPN 表达式（如 'close 5 ts_mean'）转化为 token ID 序列。
    """
    def __init__(self, base_fields: List[str] = None, const_buckets: List[float] = None):
        """
        Args:
            base_fields: 基础行情字段列表，例如 ['open', 'high', 'low', 'close', 'volume']
            const_buckets: 预定义常量列表，例如 [1, 3, 5, 10, 20]
        """
        if base_fields is None:
            base_fields = ['open', 'high', 'low', 'close', 'volume']
        if const_buckets is None:
            const_buckets = [1, 3, 5, 10, 20, 0.1, 0.5]

        self.base_fields = base_fields
        self.const_buckets = const_buckets

        # 特殊 token
        self.special_tokens = ['[PAD]', '[BOS]', '[SEP]']

        # 基础字段 token
        field_tokens = base_fields

        # 常量 token
        const_tokens = [f'CONST_{c}' for c in const_buckets]

        # 算子 token：自动扫描 operators.py
        op_tokens = []
        for name, fn in inspect.getmembers(operators, inspect.isfunction):
            if not name.startswith("_"):
                op_tokens.append(name)

        # 词表
        arith = ['+', '-', '*', '/']
        self.vocab: List[str] = self.special_tokens + field_tokens + const_tokens + op_tokens + arith

        # ID ↔ Token 映射
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}

        # 便捷属性
        self.pad_token_id = self.token_to_id['[PAD]']
        self.bos_token_id = self.token_to_id['[BOS]']
        self.sep_token_id = self.token_to_id['[SEP]']

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, expr: str, add_special_tokens: bool = True) -> List[int]:
        """
        将 RPN 表达式编码为 token ID 序列。
        """
        tokens = expr.strip().split()
        ids = []
        for tk in tokens:
            # 基础字段
            if tk in self.token_to_id:
                ids.append(self.token_to_id[tk])
            # 常量：动态加入最近的 bucket
            elif self._is_float(tk):
                val = float(tk)
                const_tok = self._map_to_const(val)
                ids.append(self.token_to_id[const_tok])
            else:
                raise ValueError(f"未知 token：{tk}")

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.sep_token_id]
        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        """
        将 token ID 序列解码为 RPN 表达式。
        """
        toks = []
        for idx in ids:
            if remove_special_tokens and idx in [self.bos_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            toks.append(self.id_to_token.get(idx, 'UNK'))
        # 常量还原
        toks = [self._const_to_str(tk) for tk in toks]
        return " ".join(toks)

    def _map_to_const(self, val: float) -> str:
        """
        将任意浮点数映射到最近的常量 bucket。
        """
        closest = min(self.const_buckets, key=lambda x: abs(x - val))
        return f'CONST_{closest}'

    def _const_to_str(self, token: str) -> str:
        """
        将 CONST_x 还原为 x 的字符串。
        """
        if token.startswith('CONST_'):
            return token.split('_', 1)[1]
        return token

    @staticmethod
    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

