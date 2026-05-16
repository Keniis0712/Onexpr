import dataclasses
from typing import Optional


@dataclasses.dataclass
class Frame:
    prev: Optional["Frame"]
    nonlocal_vars: list[str]
    global_vars: list[str]
    in_async_def: bool = False
    temp_var_num: Optional[int] = None
    loops: list[str] = dataclasses.field(default_factory=list)
    func_helper_var: Optional[str] = None
    reserved_names: Optional[set] = None  # only the top frame owns this
    exc_stack: list[str] = dataclasses.field(default_factory=list)

    def get_temp_var_num(self) -> int:
        if self.temp_var_num is None:
            self.temp_var_num = self.prev.get_temp_var_num()

        return self.temp_var_num

    def get_reserved_names(self) -> set:
        if self.reserved_names is not None:
            return self.reserved_names
        return self.prev.get_reserved_names()

    def get_temp_var(self):
        reserved = self.get_reserved_names()
        while True:
            temp_var_num = self.get_temp_var_num()
            self.temp_var_num = self.temp_var_num + 1
            name = f"temp_{temp_var_num}"
            if name not in reserved:
                return name

    def enter_loop(self):
        self.loops.append(self.get_temp_var())

    def get_cur_loop_var(self):
        assert len(self.loops)
        return self.loops[-1]

    def exit_loop(self):
        self.loops.pop()
