# tokenizer.py
import re
import inspect
import operators
from typing import List, Dict

class AlphaTokenizer:
    """
    将逆波兰表达式 (RPN) 与 token 序列相互映射的分词器。

    - **词表**：基础行情字段、常量桶、所有算子以及四则运算符  
    - **特殊标记**：`[PAD]` 填充、`[BOS]` 序列起始、`[SEP]` 序列终止
    """
    def __init__(self, base_fields: List[str] = None, const_buckets: List[float] = None):
        """
        构造分词器并自动扫描 `operators.py` 生成完整词表。

        参数
        ----------
        base_fields : list[str], 可选
            基础行情字段名称列表，默认为  
            ``['open', 'high', 'low', 'close', 'volume']``。
        const_buckets : list[float], 可选
            预定义浮点常量桶，默认为  
            ``[1, 3, 5, 10, 20, 0.1, 0.5]``。
        """
        if base_fields is None:
            base_fields = ['open', 'high', 'low', 'close', 'volume']
        if const_buckets is None:
            const_buckets = [1, 3, 5, 10, 20, 0.1, 0.5]

        self.base_fields = base_fields
        self.const_buckets = const_buckets

        self.special_tokens = ['[PAD]', '[BOS]', '[SEP]']
        field_tokens = base_fields
        const_tokens = [f'CONST_{c}' for c in const_buckets]
        op_tokens: List = list(operators.FUNC_MAP.keys())
        
        self.vocab: List[str] = self.special_tokens + field_tokens + const_tokens + op_tokens

        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}
        self.operand_type_map: dict[str, str] = {}

        for f in self.base_fields:
            self.operand_type_map[f] = "Series"
        for c in self.const_buckets:
            tok = f"CONST_{c}"
            # 先把 c 转成 float，再看是不是整数
            if float(c).is_integer():
                self.operand_type_map[tok] = "Scalar_INT"
            else:
                self.operand_type_map[tok] = "Scalar_FLOAT"

        self.pad_token_id = self.token_to_id['[PAD]']
        self.bos_token_id = self.token_to_id['[BOS]']
        self.sep_token_id = self.token_to_id['[SEP]']

    @property
    def vocab_size(self) -> int:
        """
        返回当前词表大小。

        返回
        ----------
        int
            ``len(self.vocab)``。
        """
        return len(self.vocab)

    def encode(self, expr: str, add_special_tokens: bool = True) -> List[int]:
        """
        将 RPN 字符串编码为 token ID 序列。

        参数
        ----------
        expr : str
            形如 ``"close 5 ts_mean"`` 的逆波兰表达式。
        add_special_tokens : bool, 可选
            是否在首尾分别加入 `[BOS]` 与 `[SEP]`，默认为 ``True``。

        返回
        ----------
        list[int]
            token ID 序列。
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
        将 token ID 序列还原为 RPN 字符串。

        参数
        ----------
        ids : list[int]
            编码后的 token ID 列表。
        remove_special_tokens : bool, 可选
            是否去除 `[BOS]` / `[SEP]` / `[PAD]`，默认为 ``True``。

        返回
        ----------
        str
            逆波兰表达式，常量会被还原为原始数字字符串。
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
        将任意浮点数映射到最近的常量桶，并返回其 token 名。

        参数
        ----------
        val : float
            原始浮点数。

        返回
        ----------
        str
            形如 ``"CONST_5"`` 的 token 名称。
        """
        closest = min(self.const_buckets, key=lambda x: abs(x - val))
        return f'CONST_{closest}'

    def _const_to_str(self, token: str) -> str:
        """
        将常量 token 名恢复为数字字符串。

        参数
        ----------
        token : str
            形如 ``"CONST_3"`` 的 token。

        返回
        ----------
        str
            ``"3"``；如果传入非常量 token，则原样返回。
        """
        if token.startswith('CONST_'):
            return token.split('_', 1)[1]
        return token

    @staticmethod
    def _is_float(s: str) -> bool:
        """
        判断字符串能否安全转换为 `float`。

        参数
        ----------
        s : str
            待检测字符串。

        返回
        ----------
        bool
            可转换返回 ``True``，否则 ``False``。
        """
        try:
            float(s)
            return True
        except ValueError:
            return False

