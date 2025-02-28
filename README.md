# Onexpr - Change the program to one expression


**Onexpr** is used to cover the whole python program to one single expression.
It can be just for fun, but can also be used to obfuscate the code


## Usage
**Onexpr** is a cli tool, you can run it in the shell.
```bash
$ ./onexpr.py --input my_program.py --output obfuscated.py
```

## Defect
- It still can't use`break` or the similar statements in `try` block.
