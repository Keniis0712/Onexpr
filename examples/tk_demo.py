"""Tkinter demo: a small counter + todo list app.

Covers:
- ttk widgets (Label, Button, Entry, Listbox, Frame)
- after() based scheduling
- StringVar / IntVar binding
- event callbacks closing over outer state
- exception handling on user input

Run the original:
    python examples/tk_demo.py
or onexpr-obfuscated:
    python onexpr.py --input examples/tk_demo.py --output obf.py
    python obf.py
"""

import tkinter as tk
from tkinter import ttk, messagebox


def make_app():
    root = tk.Tk()
    root.title("onexpr tk demo")
    root.geometry("360x420")

    # --- Counter (auto-incrementing every 500ms) ---
    counter_var = tk.IntVar(value=0)

    counter_frame = ttk.LabelFrame(root, text="Counter (auto)")
    counter_frame.pack(fill="x", padx=10, pady=8)

    ttk.Label(counter_frame, textvariable=counter_var, font=("", 24)).pack(pady=4)

    def tick():
        counter_var.set(counter_var.get() + 1)
        root.after(500, tick)

    root.after(500, tick)

    # --- Todo list ---
    todos_frame = ttk.LabelFrame(root, text="Todos")
    todos_frame.pack(fill="both", expand=True, padx=10, pady=8)

    entry_var = tk.StringVar()

    entry_row = ttk.Frame(todos_frame)
    entry_row.pack(fill="x", padx=6, pady=4)

    entry = ttk.Entry(entry_row, textvariable=entry_var)
    entry.pack(side="left", fill="x", expand=True)
    entry.focus_set()

    listbox = tk.Listbox(todos_frame, height=10)
    listbox.pack(fill="both", expand=True, padx=6, pady=4)

    def add_todo(event=None):
        text = entry_var.get().strip()
        if not text:
            messagebox.showwarning("empty", "todo can't be empty")
            return
        listbox.insert("end", text)
        entry_var.set("")
        status_var.set(f"{listbox.size()} item(s)")

    def remove_selected():
        sel = listbox.curselection()
        if not sel:
            return
        for i in reversed(sel):
            listbox.delete(i)
        status_var.set(f"{listbox.size()} item(s)")

    def clear_all():
        if not listbox.size():
            return
        if messagebox.askyesno("clear", "remove all?"):
            listbox.delete(0, "end")
            status_var.set("0 item(s)")

    ttk.Button(entry_row, text="Add", command=add_todo).pack(side="left", padx=4)
    entry.bind("<Return>", add_todo)

    button_row = ttk.Frame(todos_frame)
    button_row.pack(fill="x", padx=6, pady=4)
    ttk.Button(button_row, text="Remove selected", command=remove_selected).pack(
        side="left", padx=2
    )
    ttk.Button(button_row, text="Clear all", command=clear_all).pack(
        side="left", padx=2
    )

    status_var = tk.StringVar(value="0 item(s)")
    ttk.Label(root, textvariable=status_var, anchor="w").pack(
        side="bottom", fill="x", padx=10, pady=4
    )

    return root


if __name__ == "__main__":
    make_app().mainloop()
