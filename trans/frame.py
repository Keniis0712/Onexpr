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
    # Legacy return mode: emits the older `(value, True)` tuple-and-[0]
    # convention instead of going through _FuncHelper. Used for the
    # internal helper classes themselves (chicken-and-egg: their own
    # bodies can't depend on _FuncHelper because they're defining it).
    # Limitation of legacy mode: `return` inside a loop doesn't escape
    # the loop. Helper methods don't do that, so they fit fine.
    legacy_return: bool = False
    # Set when this frame is a class body (so parse_ann_assign knows to
    # write to __annotations__).
    is_class_body: bool = False
    # Names that this frame's owner FunctionDef has marked as boxed
    # (try-clause writes, nonlocal targets). parse_import / parse_for
    # / etc. that synthesize fresh Name(Store) targets at parse time
    # consult this so the synthesized assignment lands in
    # helper_var._b_<name> instead of a per-clause-lambda local that
    # would never be visible elsewhere.
    boxed_names: Optional[set] = None
    # Map from a helper's source-file name (e.g. '_FuncHelper') to the
    # name actually used in the emitted output. Equal to {} (or a
    # subset of identity mappings) when the user's code doesn't shadow
    # any helper. When the user assigns `_FuncHelper = ...` we instead
    # rename the helper to a fresh temp_N so the user's binding can't
    # break the runtime. Only the top frame owns this; descendants
    # consult via prev.
    helper_names: Optional[dict] = None
    # Map from a helper's member identifier (method or self-attribute,
    # e.g. 'do_return', 'returned', 'pending_continue', 'state',
    # '_exc_stack') to the name actually emitted. Identity map when
    # mangling is off; populated with fresh temp_N strings when
    # --replace-name global is used. Only the top frame owns it.
    helper_members: Optional[dict] = None
    # Prefix used when boxing a user-level name onto the function helper
    # for nonlocal / try-clause assigns (the `_b_` prefix in the source
    # `helper._b_x`). Becomes a fresh temp_N-style prefix under
    # mangling. Only the top frame owns it.
    helper_box_prefix: Optional[str] = None

    def get_temp_var_num(self) -> int:
        if self.temp_var_num is None:
            self.temp_var_num = self.prev.get_temp_var_num()

        return self.temp_var_num

    def get_reserved_names(self) -> set:
        if self.reserved_names is not None:
            return self.reserved_names
        return self.prev.get_reserved_names()

    def get_helper_name(self, original: str) -> str:
        """Return the actual emitted name for a helper (e.g. _FuncHelper).
        When the user's code defines a name colliding with a helper's
        source-file identifier, the top-level rename pass remaps it to
        a fresh temp_N so user assignments can't break the runtime."""
        if self.helper_names is not None:
            return self.helper_names.get(original, original)
        return self.prev.get_helper_name(original)

    def get_helper_member(self, original: str) -> str:
        """Return the actual emitted name for a helper class member or
        state-machine self-attribute (e.g. do_return, returned, _exc).
        Identity map by default; populated when --replace-name global
        is requested."""
        if self.helper_members is not None:
            return self.helper_members.get(original, original)
        return self.prev.get_helper_member(original)

    def get_box_prefix(self) -> str:
        """Prefix used to name boxed-onto-helper variables. Defaults to
        `_b_`; rewritten to a fresh temp_N-style prefix under mangling."""
        if self.helper_box_prefix is not None:
            return self.helper_box_prefix
        return self.prev.get_box_prefix()

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
