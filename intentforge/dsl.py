"""
IntentForge DSL (Domain Specific Language)

A simple, human-readable DSL for defining and executing IntentForge workflows.
Can be used from shell, scripts, or programmatically.

DSL Syntax:
-----------
# Comments start with #

# Service calls
service.action(param1="value", param2=123)

# Variables
$result = chat.send(message="Hello")

# Pipelines (chain operations)
file.analyze(filename="img.jpg") | chat.send(message=$result.description)

# Conditionals
if $result.success then chat.send(message="OK") else chat.send(message="Error")

# Loops
for item in $items do chat.send(message=$item.name)

Example DSL Script:
-------------------
# Analyze image and describe it
$analysis = file.analyze(filename="photo.jpg", image_base64=$image)
$description = file.describe(image_base64=$image)

# Send to chat
chat.send(message="Image analysis: " + $analysis.description)

# Voice command processing
$cmd = voice.process(command="Turn on lights in living room")
if $cmd.success then
    for action in $cmd.actions do
        print("Action: " + $action.type)
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .services import services

logger = logging.getLogger(__name__)


# =============================================================================
# DSL Token Types
# =============================================================================


class TokenType(Enum):
    """DSL token types"""

    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    VARIABLE = "VARIABLE"
    DOT = "DOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    EQUALS = "EQUALS"
    ASSIGN = "ASSIGN"
    PIPE = "PIPE"
    PLUS = "PLUS"
    IF = "IF"
    THEN = "THEN"
    ELSE = "ELSE"
    FOR = "FOR"
    IN = "IN"
    DO = "DO"
    END = "END"
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    COMMENT = "COMMENT"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COLON = "COLON"
    TRUE = "TRUE"
    FALSE = "FALSE"
    IMPORT = "IMPORT"
    AS = "AS"
    EXPORT = "EXPORT"
    FUNC = "FUNC"
    RETURN = "RETURN"


@dataclass
class Token:
    """DSL token"""

    type: TokenType
    value: Any
    line: int = 0
    col: int = 0


# =============================================================================
# DSL Lexer
# =============================================================================


class DSLLexer:
    """Tokenize DSL source code"""

    KEYWORDS = {
        "if": TokenType.IF,
        "then": TokenType.THEN,
        "else": TokenType.ELSE,
        "for": TokenType.FOR,
        "in": TokenType.IN,
        "do": TokenType.DO,
        "end": TokenType.END,
        "true": TokenType.TRUE,
        "false": TokenType.FALSE,
        "import": TokenType.IMPORT,
        "as": TokenType.AS,
        "export": TokenType.EXPORT,
        "func": TokenType.FUNC,
        "return": TokenType.RETURN,
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        """Tokenize the source code"""
        while self.pos < len(self.source):
            self._skip_whitespace()
            if self.pos >= len(self.source):
                break

            char = self.source[self.pos]

            # Comments
            if char == "#":
                self._read_comment()
            # Newlines
            elif char == "\n":
                self.tokens.append(Token(TokenType.NEWLINE, "\n", self.line, self.col))
                self.pos += 1
                self.line += 1
                self.col = 1
            # Strings
            elif char in ('"', "'"):
                self._read_string(char)
            # Variables ($name)
            elif char == "$":
                self._read_variable()
            # Numbers
            elif char.isdigit() or (char == "-" and self._peek().isdigit()):
                self._read_number()
            # Identifiers and keywords
            elif char.isalpha() or char == "_":
                self._read_identifier()
            # Operators
            elif char == ".":
                self.tokens.append(Token(TokenType.DOT, ".", self.line, self.col))
                self._advance()
            elif char == "(":
                self.tokens.append(Token(TokenType.LPAREN, "(", self.line, self.col))
                self._advance()
            elif char == ")":
                self.tokens.append(Token(TokenType.RPAREN, ")", self.line, self.col))
                self._advance()
            elif char == "[":
                self.tokens.append(Token(TokenType.LBRACKET, "[", self.line, self.col))
                self._advance()
            elif char == "]":
                self.tokens.append(Token(TokenType.RBRACKET, "]", self.line, self.col))
                self._advance()
            elif char == ",":
                self.tokens.append(Token(TokenType.COMMA, ",", self.line, self.col))
                self._advance()
            elif char == ":":
                self.tokens.append(Token(TokenType.COLON, ":", self.line, self.col))
                self._advance()
            elif char == "=":
                if self._peek() == "=":
                    self.tokens.append(Token(TokenType.EQUALS, "==", self.line, self.col))
                    self._advance()
                    self._advance()
                else:
                    self.tokens.append(Token(TokenType.ASSIGN, "=", self.line, self.col))
                    self._advance()
            elif char == "|":
                self.tokens.append(Token(TokenType.PIPE, "|", self.line, self.col))
                self._advance()
            elif char == "+":
                self.tokens.append(Token(TokenType.PLUS, "+", self.line, self.col))
                self._advance()
            else:
                self._advance()

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return self.tokens

    def _advance(self):
        self.pos += 1
        self.col += 1

    def _peek(self) -> str:
        if self.pos + 1 < len(self.source):
            return self.source[self.pos + 1]
        return ""

    def _skip_whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos] in " \t\r":
            self._advance()

    def _read_comment(self):
        while self.pos < len(self.source) and self.source[self.pos] != "\n":
            self._advance()
        # Skip comments, don't add to tokens

    def _read_string(self, quote: str):
        start_col = self.col
        self._advance()  # Skip opening quote
        value = ""
        while self.pos < len(self.source) and self.source[self.pos] != quote:
            if self.source[self.pos] == "\\":
                self._advance()
                if self.pos < len(self.source):
                    escape_char = self.source[self.pos]
                    if escape_char == "n":
                        value += "\n"
                    elif escape_char == "t":
                        value += "\t"
                    else:
                        value += escape_char
                    self._advance()
            else:
                value += self.source[self.pos]
                self._advance()
        self._advance()  # Skip closing quote
        self.tokens.append(Token(TokenType.STRING, value, self.line, start_col))

    def _read_variable(self):
        start_col = self.col
        self._advance()  # Skip $
        name = ""
        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or self.source[self.pos] in "_."
        ):
            name += self.source[self.pos]
            self._advance()
        self.tokens.append(Token(TokenType.VARIABLE, name, self.line, start_col))

    def _read_number(self):
        start_col = self.col
        value = ""
        if self.source[self.pos] == "-":
            value += "-"
            self._advance()
        while self.pos < len(self.source) and (
            self.source[self.pos].isdigit() or self.source[self.pos] == "."
        ):
            value += self.source[self.pos]
            self._advance()
        if "." in value:
            self.tokens.append(Token(TokenType.NUMBER, float(value), self.line, start_col))
        else:
            self.tokens.append(Token(TokenType.NUMBER, int(value), self.line, start_col))

    def _read_identifier(self):
        start_col = self.col
        value = ""
        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or self.source[self.pos] == "_"
        ):
            value += self.source[self.pos]
            self._advance()

        # Check for keywords
        token_type = self.KEYWORDS.get(value.lower(), TokenType.IDENTIFIER)
        self.tokens.append(Token(token_type, value, self.line, start_col))


# =============================================================================
# DSL AST Nodes
# =============================================================================


@dataclass
class ASTNode:
    """Base AST node"""

    pass


@dataclass
class ServiceCall(ASTNode):
    """service.action(args)"""

    service: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Assignment(ASTNode):
    """$var = expr"""

    variable: str
    value: ASTNode


@dataclass
class Variable(ASTNode):
    """$var or $var.field"""

    name: str


@dataclass
class Literal(ASTNode):
    """String, number, bool"""

    value: Any


@dataclass
class BinaryOp(ASTNode):
    """left op right"""

    left: ASTNode
    op: str
    right: ASTNode


@dataclass
class Pipeline(ASTNode):
    """expr | expr"""

    stages: list[ASTNode]


@dataclass
class IfStatement(ASTNode):
    """if cond then expr else expr"""

    condition: ASTNode
    then_branch: ASTNode
    else_branch: ASTNode | None = None


@dataclass
class ForLoop(ASTNode):
    """for item in collection do expr"""

    variable: str
    collection: ASTNode
    body: list[ASTNode]


@dataclass
class Program(ASTNode):
    """Root node containing statements"""

    statements: list[ASTNode]


@dataclass
class ImportStatement(ASTNode):
    """import "path/to/file.dsl" [as alias]"""

    path: str
    alias: str | None = None


@dataclass
class FunctionDef(ASTNode):
    """func name(params) do body end"""

    name: str
    params: list[str]
    body: list[ASTNode]


@dataclass
class FunctionCall(ASTNode):
    """name(args)"""

    name: str
    args: list[ASTNode]


@dataclass
class ReturnStatement(ASTNode):
    """return expr"""

    value: ASTNode | None = None


@dataclass
class ExportStatement(ASTNode):
    """export $var or export func"""

    name: str


# =============================================================================
# DSL Parser
# =============================================================================


class DSLParser:
    """Parse DSL tokens into AST"""

    def __init__(self, tokens: list[Token]):
        self.tokens = [t for t in tokens if t.type != TokenType.NEWLINE or t.type == TokenType.EOF]
        self.pos = 0

    def parse(self) -> Program:
        """Parse tokens into AST"""
        statements = []
        while not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        return Program(statements)

    def _current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

    def _is_at_end(self) -> bool:
        return self._current().type == TokenType.EOF

    def _advance(self) -> Token:
        token = self._current()
        self.pos += 1
        return token

    def _match(self, *types: TokenType) -> bool:
        if self._current().type in types:
            self._advance()
            return True
        return False

    def _expect(self, token_type: TokenType, message: str) -> Token:
        if self._current().type == token_type:
            return self._advance()
        raise SyntaxError(f"{message} at line {self._current().line}, got {self._current().type}")

    def _peek_next(self) -> Token | None:
        """Look at the next token without advancing"""
        next_pos = self.pos + 1
        if next_pos < len(self.tokens):
            return self.tokens[next_pos]
        return None

    def _parse_statement(self) -> ASTNode | None:
        """Parse a single statement"""
        # Skip empty lines
        while self._match(TokenType.NEWLINE):
            pass

        if self._is_at_end():
            return None

        # Import statement: import "path" [as alias]
        if self._current().type == TokenType.IMPORT:
            return self._parse_import()

        # Function definition: func name(params) do body end
        if self._current().type == TokenType.FUNC:
            return self._parse_func()

        # Export statement: export name
        if self._current().type == TokenType.EXPORT:
            return self._parse_export()

        # Return statement: return [expr]
        if self._current().type == TokenType.RETURN:
            return self._parse_return()

        # Assignment: $var = expr (check for = after variable)
        if self._current().type == TokenType.VARIABLE:
            # Look ahead to see if this is an assignment or just a variable reference
            if self._peek_next() and self._peek_next().type == TokenType.ASSIGN:
                return self._parse_assignment()
            else:
                # Just a variable expression (e.g., $response to print it)
                return self._parse_expression()

        # If statement
        if self._current().type == TokenType.IF:
            return self._parse_if()

        # For loop
        if self._current().type == TokenType.FOR:
            return self._parse_for()

        # Expression (service call, etc.)
        return self._parse_expression()

    def _parse_import(self) -> ImportStatement:
        """Parse import "path" [as alias]"""
        self._advance()  # Skip 'import'
        path_token = self._expect(TokenType.STRING, "Expected import path string")
        alias = None
        if self._current().type == TokenType.AS:
            self._advance()  # Skip 'as'
            alias_token = self._expect(TokenType.IDENTIFIER, "Expected alias name")
            alias = alias_token.value
        return ImportStatement(path_token.value, alias)

    def _parse_func(self) -> FunctionDef:
        """Parse func name(params) do body end"""
        self._advance()  # Skip 'func'
        name_token = self._expect(TokenType.IDENTIFIER, "Expected function name")
        self._expect(TokenType.LPAREN, "Expected '(' after function name")

        params = []
        if self._current().type != TokenType.RPAREN:
            # Parse parameter list
            param = self._expect(TokenType.IDENTIFIER, "Expected parameter name")
            params.append(param.value)
            while self._match(TokenType.COMMA):
                param = self._expect(TokenType.IDENTIFIER, "Expected parameter name")
                params.append(param.value)

        self._expect(TokenType.RPAREN, "Expected ')' after parameters")
        self._expect(TokenType.DO, "Expected 'do' after function parameters")

        body = []
        while not self._is_at_end() and self._current().type != TokenType.END:
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)

        self._match(TokenType.END)
        return FunctionDef(name_token.value, params, body)

    def _parse_export(self) -> ExportStatement:
        """Parse export name"""
        self._advance()  # Skip 'export'
        if (
            self._current().type == TokenType.VARIABLE
            or self._current().type == TokenType.IDENTIFIER
        ):
            name = self._advance().value
        else:
            raise SyntaxError(
                f"Expected variable or function name after export at line {self._current().line}"
            )
        return ExportStatement(name)

    def _parse_return(self) -> ReturnStatement:
        """Parse return [expr]"""
        self._advance()  # Skip 'return'
        value = None
        if not self._is_at_end() and self._current().type not in (
            TokenType.END,
            TokenType.NEWLINE,
            TokenType.EOF,
        ):
            value = self._parse_expression()
        return ReturnStatement(value)

    def _parse_assignment(self) -> Assignment:
        """Parse $var = expr"""
        var_token = self._advance()
        self._expect(TokenType.ASSIGN, "Expected '='")
        value = self._parse_expression()
        return Assignment(var_token.value, value)

    def _parse_if(self) -> IfStatement:
        """Parse if cond then expr [else expr] [end]"""
        self._advance()  # Skip 'if'
        condition = self._parse_expression()
        self._expect(TokenType.THEN, "Expected 'then'")
        then_branch = self._parse_expression()
        else_branch = None
        if self._match(TokenType.ELSE):
            else_branch = self._parse_expression()
        self._match(TokenType.END)
        return IfStatement(condition, then_branch, else_branch)

    def _parse_for(self) -> ForLoop:
        """Parse for item in collection do body end"""
        self._advance()  # Skip 'for'
        var_token = self._expect(TokenType.IDENTIFIER, "Expected variable name")
        self._expect(TokenType.IN, "Expected 'in'")
        collection = self._parse_expression()
        self._expect(TokenType.DO, "Expected 'do'")

        body = []
        while not self._is_at_end() and self._current().type != TokenType.END:
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)

        self._match(TokenType.END)
        return ForLoop(var_token.value, collection, body)

    def _parse_expression(self) -> ASTNode:
        """Parse expression with pipeline support"""
        left = self._parse_binary()

        # Pipeline: expr | expr
        if self._match(TokenType.PIPE):
            stages = [left]
            while True:
                stages.append(self._parse_binary())
                if not self._match(TokenType.PIPE):
                    break
            return Pipeline(stages)

        return left

    def _parse_binary(self) -> ASTNode:
        """Parse binary operations (+ for string concat)"""
        left = self._parse_primary()

        while self._match(TokenType.PLUS):
            right = self._parse_primary()
            left = BinaryOp(left, "+", right)

        return left

    def _parse_primary(self) -> ASTNode:
        """Parse primary expressions"""
        token = self._current()

        # Literals
        if token.type == TokenType.STRING:
            self._advance()
            return Literal(token.value)

        if token.type == TokenType.NUMBER:
            self._advance()
            return Literal(token.value)

        if token.type == TokenType.TRUE:
            self._advance()
            return Literal(True)

        if token.type == TokenType.FALSE:
            self._advance()
            return Literal(False)

        # Variable
        if token.type == TokenType.VARIABLE:
            self._advance()
            return Variable(token.value)

        # Service call: service.action(args)
        if token.type == TokenType.IDENTIFIER:
            return self._parse_service_call()

        # Parenthesized expression
        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')'")
            return expr

        raise SyntaxError(f"Unexpected token: {token.type} at line {token.line}")

    def _parse_service_call(self) -> ServiceCall:
        """Parse service.action(args)"""
        service = self._advance().value

        self._expect(TokenType.DOT, "Expected '.'")
        action = self._expect(TokenType.IDENTIFIER, "Expected action name").value

        args = {}
        if self._match(TokenType.LPAREN):
            if self._current().type != TokenType.RPAREN:
                args = self._parse_args()
            self._expect(TokenType.RPAREN, "Expected ')'")

        return ServiceCall(service, action, args)

    def _parse_args(self) -> dict[str, Any]:
        """Parse function arguments: name=value, name=value"""
        args = {}

        while True:
            name = self._expect(TokenType.IDENTIFIER, "Expected argument name").value
            self._expect(TokenType.ASSIGN, "Expected '='")
            value = self._parse_expression()
            args[name] = value

            if not self._match(TokenType.COMMA):
                break

        return args


# =============================================================================
# DSL Interpreter
# =============================================================================


class DSLInterpreter:
    """Execute DSL AST"""

    def __init__(self, base_path: str = "."):
        self.variables: dict[str, Any] = {}
        self.functions: dict[str, FunctionDef] = {}
        self.exports: set[str] = set()
        self.imported_modules: dict[str, DSLInterpreter] = {}
        self.last_result: Any = None
        self.base_path = base_path

    async def execute(self, program: Program) -> Any:
        """Execute a DSL program"""
        result = None
        for stmt in program.statements:
            result = await self._eval(stmt)
            self.last_result = result
        return result

    async def _handle_import(self, node: ImportStatement) -> None:
        """Handle import statement"""
        import_path = node.path

        # Resolve path relative to base_path
        if not os.path.isabs(import_path):
            import_path = os.path.join(self.base_path, import_path)

        # Add .dsl extension if not present
        if not import_path.endswith(".dsl"):
            import_path += ".dsl"

        if not os.path.exists(import_path):
            raise FileNotFoundError(f"Import file not found: {import_path}")

        # Read and parse imported file
        with open(import_path, encoding="utf-8") as f:
            source = f.read()

        # Create new interpreter for imported module
        module_interpreter = DSLInterpreter(base_path=os.path.dirname(import_path))
        lexer = DSLLexer(source)
        tokens = lexer.tokenize()
        parser = DSLParser(tokens)
        program = parser.parse()

        # Execute imported module
        await module_interpreter.execute(program)

        # Store module reference
        alias = node.alias or os.path.splitext(os.path.basename(import_path))[0]
        self.imported_modules[alias] = module_interpreter

        # Import exported variables and functions
        for name in module_interpreter.exports:
            if name in module_interpreter.variables:
                self.variables[f"{alias}.{name}"] = module_interpreter.variables[name]
            if name in module_interpreter.functions:
                self.functions[f"{alias}.{name}"] = module_interpreter.functions[name]

        # Also make all exports available directly if no alias conflict
        for name in module_interpreter.exports:
            if name not in self.variables and name in module_interpreter.variables:
                self.variables[name] = module_interpreter.variables[name]
            if name not in self.functions and name in module_interpreter.functions:
                self.functions[name] = module_interpreter.functions[name]

    async def _call_function(self, node: FunctionCall) -> Any:
        """Call a user-defined function"""
        func = self.functions.get(node.name)
        if func is None:
            raise ValueError(f"Unknown function: {node.name}")

        # Save current variables
        saved_vars = self.variables.copy()

        # Bind arguments to parameters
        if len(node.args) != len(func.params):
            raise ValueError(
                f"Function {node.name} expects {len(func.params)} arguments, got {len(node.args)}"
            )

        for param, arg in zip(func.params, node.args, strict=False):
            self.variables[param] = await self._eval(arg)

        # Execute function body
        result = None
        for stmt in func.body:
            result = await self._eval(stmt)
            if isinstance(stmt, ReturnStatement):
                break

        # Restore variables (keep only function-level changes to globals)
        self.variables = saved_vars

        return result

    async def _eval(self, node: ASTNode) -> Any:
        """Evaluate an AST node"""
        if isinstance(node, Literal):
            return node.value

        if isinstance(node, Variable):
            return self._get_variable(node.name)

        if isinstance(node, BinaryOp):
            left = await self._eval(node.left)
            right = await self._eval(node.right)
            if node.op == "+":
                return str(left) + str(right)
            raise ValueError(f"Unknown operator: {node.op}")

        if isinstance(node, ServiceCall):
            return await self._call_service(node)

        if isinstance(node, Assignment):
            value = await self._eval(node.value)
            self.variables[node.variable] = value
            return value

        if isinstance(node, Pipeline):
            result = None
            for stage in node.stages:
                if isinstance(stage, ServiceCall) and result is not None:
                    # Pass previous result as context
                    stage.args["_previous"] = Literal(result)
                result = await self._eval(stage)
            return result

        if isinstance(node, IfStatement):
            condition = await self._eval(node.condition)
            if self._is_truthy(condition):
                return await self._eval(node.then_branch)
            elif node.else_branch:
                return await self._eval(node.else_branch)
            return None

        if isinstance(node, ForLoop):
            collection = await self._eval(node.collection)
            if not isinstance(collection, (list, tuple)):
                collection = [collection]

            results = []
            for item in collection:
                self.variables[node.variable] = item
                for stmt in node.body:
                    result = await self._eval(stmt)
                    results.append(result)
            return results

        if isinstance(node, ImportStatement):
            return await self._handle_import(node)

        if isinstance(node, FunctionDef):
            # Store function definition
            self.functions[node.name] = node
            return None

        if isinstance(node, FunctionCall):
            return await self._call_function(node)

        if isinstance(node, ReturnStatement):
            if node.value:
                return await self._eval(node.value)
            return None

        if isinstance(node, ExportStatement):
            # Mark variable/function as exported
            self.exports.add(node.name)
            return None

        if isinstance(node, Program):
            return await self.execute(node)

        raise ValueError(f"Unknown node type: {type(node)}")

    def _get_variable(self, name: str) -> Any:
        """Get variable value, supporting dot notation"""
        parts = name.split(".")
        value = self.variables.get(parts[0])

        for part in parts[1:]:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

        return value

    def _is_truthy(self, value: Any) -> bool:
        """Check if value is truthy"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return value.get("success", bool(value))
        return bool(value)

    async def _call_service(self, node: ServiceCall) -> Any:
        """Call a service method"""
        service = services.get(node.service)
        if service is None:
            raise ValueError(f"Unknown service: {node.service}")

        method = getattr(service, node.action, None)
        if method is None:
            raise ValueError(f"Unknown action '{node.action}' for service '{node.service}'")

        # Evaluate arguments
        kwargs = {}
        for key, value in node.args.items():
            if key == "_previous":
                continue
            if isinstance(value, ASTNode):
                kwargs[key] = await self._eval(value)
            else:
                kwargs[key] = value

        # Call method
        import inspect

        result = method(**kwargs)
        if inspect.isawaitable(result):
            result = await result

        return result


# =============================================================================
# DSL Runner (High-level API)
# =============================================================================


class DSLRunner:
    """High-level DSL execution API"""

    def __init__(self, base_path: str = "."):
        self.interpreter = DSLInterpreter(base_path=base_path)
        self.base_path = base_path

    def parse(self, source: str) -> Program:
        """Parse DSL source code"""
        lexer = DSLLexer(source)
        tokens = lexer.tokenize()
        parser = DSLParser(tokens)
        return parser.parse()

    async def run(self, source: str) -> Any:
        """Parse and execute DSL source code"""
        program = self.parse(source)
        return await self.interpreter.execute(program)

    async def run_file(self, filepath: str) -> Any:
        """Run a DSL file"""
        filepath = os.path.abspath(filepath)
        self.interpreter.base_path = os.path.dirname(filepath)
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        return await self.run(source)

    def run_sync(self, source: str) -> Any:
        """Synchronous version of run"""
        return asyncio.run(self.run(source))

    def run_file_sync(self, filepath: str) -> Any:
        """Synchronous version of run_file"""
        return asyncio.run(self.run_file(filepath))

    def get_variables(self) -> dict[str, Any]:
        """Get all variables"""
        return self.interpreter.variables.copy()

    def get_functions(self) -> dict[str, FunctionDef]:
        """Get all defined functions"""
        return self.interpreter.functions.copy()

    def set_variable(self, name: str, value: Any):
        """Set a variable"""
        self.interpreter.variables[name] = value


# =============================================================================
# DSL Debugger
# =============================================================================


class DebugEvent:
    """Debug event types"""

    STEP = "step"
    BREAKPOINT = "breakpoint"
    VARIABLE_CHANGE = "variable_change"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class DebugState:
    """Current debug state"""

    line: int = 0
    statement_index: int = 0
    current_node: ASTNode | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    call_stack: list[str] = field(default_factory=list)
    paused: bool = False
    finished: bool = False
    last_result: Any = None


class DSLDebugger:
    """
    DSL Debugger with step-through execution and variable inspection.

    Usage:
        debugger = DSLDebugger(source_code)
        debugger.set_breakpoint(5)  # Line 5

        async for event in debugger.run():
            print(f"Line {event.line}: {event.variables}")
            if event.paused:
                # Inspect variables
                print(debugger.get_variables())
                # Continue
                debugger.step()
    """

    def __init__(self, source: str, base_path: str = "."):
        self.source = source
        self.source_lines = source.split("\n")
        self.base_path = base_path

        # Parse program
        lexer = DSLLexer(source)
        self.tokens = lexer.tokenize()
        parser = DSLParser(self.tokens)
        self.program = parser.parse()

        # Debug state
        self.state = DebugState()
        self.breakpoints: set[int] = set()
        self.step_mode = False
        self._continue_event = asyncio.Event()
        self._continue_event.set()  # Not paused initially

        # Interpreter
        self.interpreter = DSLInterpreter(base_path=base_path)

        # Event callbacks
        self._on_step_callbacks: list[callable] = []

    def set_breakpoint(self, line: int) -> None:
        """Set a breakpoint at line number"""
        self.breakpoints.add(line)

    def remove_breakpoint(self, line: int) -> None:
        """Remove a breakpoint"""
        self.breakpoints.discard(line)

    def clear_breakpoints(self) -> None:
        """Clear all breakpoints"""
        self.breakpoints.clear()

    def get_breakpoints(self) -> list[int]:
        """Get all breakpoints"""
        return sorted(self.breakpoints)

    def step(self) -> None:
        """Step to next statement"""
        self.step_mode = True
        self._continue_event.set()

    def continue_execution(self) -> None:
        """Continue execution until next breakpoint"""
        self.step_mode = False
        self._continue_event.set()

    def pause(self) -> None:
        """Pause execution"""
        self.state.paused = True
        self._continue_event.clear()

    def get_variables(self) -> dict[str, Any]:
        """Get current variables"""
        return self.interpreter.variables.copy()

    def get_functions(self) -> list[str]:
        """Get defined function names"""
        return list(self.interpreter.functions.keys())

    def get_source_line(self, line: int) -> str:
        """Get source code at line"""
        if 0 < line <= len(self.source_lines):
            return self.source_lines[line - 1]
        return ""

    def get_state(self) -> DebugState:
        """Get current debug state"""
        self.state.variables = self.get_variables()
        return self.state

    def on_step(self, callback: callable) -> None:
        """Register callback for step events"""
        self._on_step_callbacks.append(callback)

    async def _notify_step(self, node: ASTNode, line: int) -> None:
        """Notify listeners of step event"""
        self.state.line = line
        self.state.current_node = node
        self.state.variables = self.get_variables()

        for callback in self._on_step_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(self.state)
            else:
                callback(self.state)

    async def _check_breakpoint(self, line: int) -> bool:
        """Check if we should pause at this line"""
        if self.step_mode or line in self.breakpoints:
            self.state.paused = True
            self._continue_event.clear()
            await self._continue_event.wait()
            self.state.paused = False
            return True
        return False

    async def run(self) -> Any:
        """Run the program with debugging"""
        self.state = DebugState()
        result = None

        for i, stmt in enumerate(self.program.statements):
            self.state.statement_index = i

            # Get line number from token if available
            line = getattr(stmt, "line", i + 1) if hasattr(stmt, "line") else i + 1

            # Notify step
            await self._notify_step(stmt, line)

            # Check breakpoint
            await self._check_breakpoint(line)

            # Execute statement
            try:
                result = await self.interpreter._eval(stmt)
                self.state.last_result = result
            except Exception:
                self.state.finished = True
                raise

        self.state.finished = True
        return result

    async def run_interactive(self):
        """
        Generator for interactive debugging.

        Yields debug states at each step.
        """
        self.state = DebugState()
        self.step_mode = True  # Start in step mode

        for i, stmt in enumerate(self.program.statements):
            self.state.statement_index = i
            line = i + 1

            # Pause and yield state
            self.state.line = line
            self.state.current_node = stmt
            self.state.variables = self.get_variables()
            self.state.paused = True

            yield self.state

            # Execute statement
            try:
                result = await self.interpreter._eval(stmt)
                self.state.last_result = result
            except Exception:
                self.state.finished = True
                yield self.state
                raise

        self.state.finished = True
        self.state.paused = False
        yield self.state

    def run_sync(self) -> Any:
        """Synchronous run (no breakpoints)"""
        return asyncio.run(self.run())

    def __repr__(self) -> str:
        return f"DSLDebugger(lines={len(self.source_lines)}, breakpoints={self.breakpoints})"


# =============================================================================
# DSL to Python/Shell Code Generator
# =============================================================================


class DSLCodeGenerator:
    """Generate executable code from DSL"""

    def to_python(self, program: Program) -> str:
        """Generate Python code from DSL AST"""
        lines = [
            "#!/usr/bin/env python3",
            '"""Auto-generated from IntentForge DSL"""',
            "",
            "import asyncio",
            "from intentforge.services import services",
            "",
            "",
            "async def main():",
        ]

        for stmt in program.statements:
            code = self._stmt_to_python(stmt, indent=1)
            lines.append(code)

        lines.extend(
            [
                "",
                "",
                'if __name__ == "__main__":',
                "    asyncio.run(main())",
            ]
        )

        return "\n".join(lines)

    def to_shell(self, program: Program) -> str:
        """Generate shell script from DSL AST"""
        lines = [
            "#!/bin/bash",
            "# Auto-generated from IntentForge DSL",
            "",
            "set -e",
            "",
        ]

        for stmt in program.statements:
            code = self._stmt_to_shell(stmt)
            lines.append(code)

        return "\n".join(lines)

    def _stmt_to_python(self, node: ASTNode, indent: int = 0) -> str:
        """Convert AST node to Python code"""
        prefix = "    " * indent

        if isinstance(node, ServiceCall):
            args_str = ", ".join(f"{k}={self._value_to_python(v)}" for k, v in node.args.items())
            return f'{prefix}await services.get("{node.service}").{node.action}({args_str})'

        if isinstance(node, Assignment):
            value = self._stmt_to_python(node.value, 0).strip()
            return f"{prefix}{node.variable} = {value}"

        if isinstance(node, Variable):
            return f"{prefix}{node.name}"

        if isinstance(node, Literal):
            return f"{prefix}{node.value!r}"

        if isinstance(node, BinaryOp):
            left = self._stmt_to_python(node.left, 0).strip()
            right = self._stmt_to_python(node.right, 0).strip()
            return f"{prefix}str({left}) + str({right})"

        if isinstance(node, IfStatement):
            cond = self._stmt_to_python(node.condition, 0).strip()
            then_code = self._stmt_to_python(node.then_branch, indent + 1)
            code = f"{prefix}if {cond}:\n{then_code}"
            if node.else_branch:
                else_code = self._stmt_to_python(node.else_branch, indent + 1)
                code += f"\n{prefix}else:\n{else_code}"
            return code

        if isinstance(node, ForLoop):
            coll = self._stmt_to_python(node.collection, 0).strip()
            body = "\n".join(self._stmt_to_python(s, indent + 1) for s in node.body)
            return f"{prefix}for {node.variable} in {coll}:\n{body}"

        return f"{prefix}# Unknown node: {type(node)}"

    def _value_to_python(self, value: Any) -> str:
        """Convert value to Python literal"""
        if isinstance(value, Literal):
            return repr(value.value)
        if isinstance(value, Variable):
            return value.name
        if isinstance(value, ASTNode):
            return self._stmt_to_python(value, 0).strip()
        return repr(value)

    def _stmt_to_shell(self, node: ASTNode) -> str:
        """Convert AST node to shell command"""
        if isinstance(node, ServiceCall):
            args_json = json.dumps(
                {
                    k: v.value
                    if isinstance(v, Literal)
                    else f"${v.name}"
                    if isinstance(v, Variable)
                    else str(v)
                    for k, v in node.args.items()
                }
            )
            return f"intentforge dsl-call {node.service} {node.action} '{args_json}'"

        if isinstance(node, Assignment):
            # Shell variable assignment
            return f'{node.variable}=$(intentforge dsl-eval "{self._node_to_dsl(node.value)}")'

        return f"# {node}"

    def _node_to_dsl(self, node: ASTNode) -> str:
        """Convert AST node back to DSL string"""
        if isinstance(node, ServiceCall):
            args = ", ".join(
                f'{k}="{v.value}"' if isinstance(v, Literal) else f"{k}=${v.name}"
                for k, v in node.args.items()
            )
            return f"{node.service}.{node.action}({args})"
        if isinstance(node, Literal):
            return repr(node.value)
        if isinstance(node, Variable):
            return f"${node.name}"
        return str(node)


# =============================================================================
# Built-in DSL Functions
# =============================================================================


BUILTIN_FUNCTIONS = {
    "print": lambda *args: print(*args),
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "json": json.dumps,
    "parse_json": json.loads,
}


# =============================================================================
# CLI Integration
# =============================================================================


def run_dsl_file(filepath: str) -> Any:
    """Run a DSL file"""
    with open(filepath) as f:
        source = f.read()
    runner = DSLRunner()
    return runner.run_sync(source)


def run_dsl_command(command: str) -> Any:
    """Run a single DSL command"""
    runner = DSLRunner()
    return runner.run_sync(command)


def generate_python_from_dsl(source: str) -> str:
    """Generate Python code from DSL"""
    runner = DSLRunner()
    program = runner.parse(source)
    generator = DSLCodeGenerator()
    return generator.to_python(program)


def generate_shell_from_dsl(source: str) -> str:
    """Generate shell script from DSL"""
    runner = DSLRunner()
    program = runner.parse(source)
    generator = DSLCodeGenerator()
    return generator.to_shell(program)
