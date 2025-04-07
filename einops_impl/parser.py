from typing import List, Dict, Tuple, Set, Optional
import re

class Parser:
    def __init__(self,pattern: str):
        self.pattern = pattern
        self.input_pattern = ''
        self.output_pattern = ''
        self.axes_names: Set[str] = set()
        self.grouped_axes: Dict[str, List[str]] = {}

    def _validate_single_ellipsis(self, pattern: str):
        """Validate that pattern contains at most one ellipsis"""
        ellipsis_count = pattern.count('...')
        if ellipsis_count > 1:
            raise ValueError("Pattern must contain at most one ellipsis")

    def parse(self):
        if '->' not in self.pattern:
            raise ValueError("Pattern must contain '->'")
        
        self.input_pattern, self.output_pattern = self.pattern.split('->')

        self._validate_single_ellipsis(self.input_pattern)
        self._validate_single_ellipsis(self.output_pattern)


        print(f"Debug input:{self.input_pattern} and output:{self.output_pattern}")

        input_axes = self._parse_expression(self.input_pattern)
        output_axes = self._parse_expression(self.output_pattern)

        return input_axes, output_axes
    
    def _find_matching_parenthesis(self, expression: str, start: int) -> int:
        """Find the matching closing parenthesis from start position"""
        count = 1
        i = start + 1
        while i < len(expression) and count > 0:
            if expression[i] == '(':
                count += 1
            elif expression[i] == ')':
                count -= 1
            i += 1
            
        if count != 0:
            raise ValueError("Unmatched parentheses in expression")
            
        return i
    
    def _parse_expression(self, expression: str, start: int = 0, end: Optional[int] = None) -> List[str]:        
        result = []
        i = start

        if end is None:
            end = len(expression)


        while i < end:
            char = expression[i]

            if char == '.' and expression[i:i+3] == '...':
                result.append('...')
                i += 3
                continue
            elif char == '(':
                j = self._find_matching_parenthesis(expression,i)

                group_content = self._parse_expression(expression,i+1,j-1)

                group_name = f"group_{len(self.grouped_axes)}"

                self.grouped_axes[group_name] = group_content

                result.append(group_name)

                i = j
            elif char.isalpha() or char.isdigit():
                j = i
                while(j < end and (expression[j].isalnum() or expression[j] == '_' or expression[j].isdigit())):
                    j += 1
                
                axis_name = expression[i:j]
                self.axes_names.add(axis_name)
                result.append(axis_name)

                i = j
            elif char.isspace():
                i+=1
            
            else:
                raise ValueError(f"Invalid character in expression {char}")
            
        return result
    

                