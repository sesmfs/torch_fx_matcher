import torch
import torch.nn as nn
import torch.fx
from copy import deepcopy
import re

class Node:
    def __init__(self, name, type, idd):
        self.names = name if isinstance(name, list) else [name]
        self.args  = []
        self.users = []
        self.type  = type
        self.idd   = idd

    def target(self):
        return '/'.join(self.names) + f"_{self.idd}"

    def __repr__(self):
        argnames = ",".join([item.target() for item in self.args])
        usernames = ",".join([item.target() for item in self.users])
        return f"{self.target()}[{self.type}]({argnames})->({usernames})"

class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.variables = {}

    def __repr__(self):
        output = [f"Graph: {len(self.nodes)} nodes, {len(self.inputs)} inputs, {len(self.outputs)} outputs"]
        output.append(f"Inputs:")
        for node in self.inputs:
            output.append(f"\t{node}")

        output.append("")
        output.append(f"Body:")
        for node in self.nodes:
            output.append(f"\t{node}")

        output.append("")
        output.append(f"Outputs:")
        for node in self.outputs:
            output.append(f"\t{node}")
        return "\n".join(output)

    def add_node(self, name, args, outputs, idd):
        
        node = Node(name, "module", idd)
        node_args = []
        node_users = []
        for arg in args:
            if arg != "?" and arg not in self.variables:
                raise KeyError(f"Undefined argument: [{arg}]")

            if arg == "?":
                new_node = Node("?", "placeholder", idd)
                new_node.users.append(node)
                self.inputs.append(new_node)
                node_args.append(new_node)
                continue
            
            parent = self.variables[arg]
            node_args.append(parent)
            i = parent.users.index(arg)
            parent.users[i] = node

        for user in outputs:
            if user == "?":
                new_node = Node("?", "output", idd)
                new_node.args.append(node)
                self.outputs.append(new_node)
                node_users.append(new_node)
                continue
                
            node_users.append(user)
            self.variables[user] = node
        
        node.args = node_args
        node.users = node_users
        self.nodes.append(node)

class Lexer:
    def __init__(self, pattern):
        
        # Compile the extraction regular expression.
        extract_name_and_argument = re.compile("([\W\w]+)\(([\W\w]+)\)")
        # Remove spaces and split patterns by the break line.
        lines = [item.strip() for item in pattern.replace(" ", "").split("\n")]

        # Parsing patterns by lexical analyzer.
        self.pattern  = pattern
        self.lines    = lines
        self.graph    = Graph()
        for iline, line in enumerate(lines):
            if line.startswith("#") or line == "":
                continue

            names_and_arguments = extract_name_and_argument.findall(line)
            
            assert len(names_and_arguments) == 1, f"Unexpected line: {line}. The valid symbol is: name(input_argument, output_argument)"
            operator_names, argumants = names_and_arguments[0]
            inputs, outputs = self.parse_arguments(argumants)
            self.graph.add_node(operator_names.split("/"), inputs, outputs, iline)

    def parse_variable(self):
        
        variable_name = ""
        while self.itoken < len(self.symbols):
            self.token = self.symbols[self.itoken]
            
            # If a valid token(alpha/number/_ or ?) for variable.
            if self.token.isalnum() or self.token == "?" or self.token == "_":
                variable_name += self.token
            else:
                break
            
            self.itoken += 1
        return variable_name
    
    def parse_list(self):
        self.itoken += 1
        lists = [self.parse_variable()]
        while self.itoken < len(self.symbols):
            self.token = self.symbols[self.itoken]
            if self.token == ",":
                self.itoken += 1
                name = self.parse_variable()
                lists.append(name)
                continue
            elif self.token == "]":
                self.itoken += 1
                break
            else:
                raise ValueError(f"Unexpected token: {self.token}")
        assert self.token == "]", f"Unexpected end token for list: ], pos: {self.itoken}"
        return lists
        
    def parse_arguments(self, symbols):
        self.itoken = 0
        self.symbols = symbols
        
        lists = []
        while self.itoken < len(symbols):
            self.token = symbols[self.itoken]
            if self.token == "[":
                lists.append(self.parse_list())
            else:
                lists.append([self.parse_variable()])      
            self.itoken += 1
        assert len(lists) == 2, f"Unexpected number of params: {len(lists)}"
        return lists


class Matcher:
    def __init__(self):
        self.matched = []
        self.traced  = None
        self.modules = dict()

    @staticmethod
    def _replace_node_module(node: torch.fx.Node, modules, new_module: torch.nn.Module):
        assert(isinstance(node.target, str))
        *parent, name = node.target.rsplit('.', 1)
        parent_name, name = parent[0] if parent else '', name
        modules[node.target] = new_module
        setattr(modules[parent_name], name, new_module)

    @staticmethod
    def _fuse_conv_bn(conv:nn.Conv2d, bn:nn.BatchNorm2d):
        conv = deepcopy(conv)
        w = conv.weight.data
        b = 0 if conv.bias is None else conv.bias.data
        running_mean = bn.running_mean
        running_std  = torch.rsqrt(bn.running_var + bn.eps)
        gamma        = bn.weight.data
        beta         = bn.bias.data
        conv.weight.data = w * (gamma * running_std).view(-1, 1, 1, 1)
        b                = (b - running_mean) * (gamma * running_std) + beta
        if conv.bias is None:
            conv.register_parameter("bias", nn.Parameter(b))
        else:
            conv.bias.data = b
        return conv

    def _match(self, condition, anchor):
        if "?" not in condition.names and self._node_module_name(anchor) not in condition.names:
            return False
        
        all_inputs_is_placeholder  = all([item.type == "placeholder" for item in condition.args])
        all_outputs_is_placeholder = all([item.type == "output" for item in condition.users])
        if not all_inputs_is_placeholder:
            if len(condition.args) != len(anchor.args):
                return False

        if not all_outputs_is_placeholder:
            if len(condition.users) != len(anchor.users):
                return False

        if not all_inputs_is_placeholder:
            for c_arg, a_arg in zip(condition.args, anchor.args):
                if c_arg.type == "placeholder":
                    continue
                    
                if self._node_module_name(a_arg) not in c_arg.names:
                    continue
        return True
    
    def _node_module_name(self, node):
        if node.op == "call_module":
            module = self.modules[node.target]
            return module.__class__.__name__
        elif node.op == "call_function":
            return node.target.__name__
        elif node.op == "placeholder" or node.op == "output":
            return node.target
    
    def _try_to_match(self, anchor):
        matched_paths = []
        params_stack = [[[anchor], self.lexer.graph.nodes[0]]]
        while len(params_stack) > 0:
            path, condition = params_stack.pop()
            if condition.type == "output":
                matched_paths.append(path[:-1])
                continue

            anchor = path[-1]
            if not self._match(condition, anchor):
                continue
                
            all_outputs_is_placeholder = all([item.type == "output" for item in condition.users])
            for i, output_user in enumerate(anchor.users):
                if all_outputs_is_placeholder:
                    params_stack.append([path + [output_user], condition.users[0]])
                else:
                    params_stack.append([path + [output_user], condition.users[i]])

        return matched_paths
    
    def get_module(self, node):
        if node.op == "call_module":
            return self.modules[node.target]
        elif node.op == "call_function":
            return node.target
        else:
            return None

    def fuse_bn(self):
        def _fuse_bn_impl(matcher, isubgraph, subgraph):
            conv = subgraph[0]
            bn   = subgraph[1]
            fused_conv = Matcher._fuse_conv_bn(matcher.get_module(conv), matcher.get_module(bn))
            Matcher._replace_node_module(conv, matcher.modules, fused_conv)
            bn.replace_all_uses_with(conv)
            matcher.traced.graph.erase_node(bn)

        self.match("Conv2d(?, x1) \n BatchNorm2d(x1, ?)")
        self.replace(_fuse_bn_impl)
        return self

    def trace(self, model):
        self.traced: torch.fx.GraphModule = torch.fx.symbolic_trace(model)
        self.modules = dict(self.traced.named_modules())
        return self

    def match(self, pattern):
        self.lexer = Lexer(pattern)
        all_matched_pairs = []
        for node in self.traced.graph.nodes:
            all_matched_pairs.extend(self._try_to_match(node))
        self.matched = all_matched_pairs
        return self
    
    def print_matchs(self):
        print("=====================================================================")
        print(f"Found {len(self.matched)} subgraphs:")
        for i, subgraph in enumerate(self.matched):
            args_names = lambda item: ",".join([str(item) for item in item.args])
            subgraph_names = "\n\t\t".join([f"{item.name}: {self._node_module_name(item)}({args_names(item)})->{list(item.users.keys())}" for item in subgraph])
            print(f"\tSubgraph{i}: \n\t\t{subgraph_names}")
            
        pattern_text = "\n\t".join(self.lexer.lines)
        print(f"\nPattern is:\n\t{pattern_text}")
        print("=====================================================================")
        return self
    
    # replace some subgraph to new
    def replace(self, new_graph_fn=None):
        for i, subgraph in enumerate(self.matched):
            new_graph_fn(self, i, subgraph)

        self.recompile()
        return self

    def recompile(self):
        self.traced.graph.lint()
        self.traced.recompile()
        self.modules = dict(self.traced.named_modules())
        return self
