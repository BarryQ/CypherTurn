"""Property graph schema definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class DataType(Enum):
    STR = "string"
    INT = "integer"
    FLOAT = "float"
    BOOL = "boolean"


@dataclass
class EntitySchema:
    label: str
    properties: Dict[str, DataType] = field(default_factory=dict)
    description: str = ""


@dataclass
class RelationSchema:
    label: str
    subj_label: str
    obj_label: str
    properties: Dict[str, DataType] = field(default_factory=dict)
    description: str = ""


@dataclass
class PropertyGraphSchema:
    name: str
    entities: List[EntitySchema] = field(default_factory=list)
    relations: List[RelationSchema] = field(default_factory=list)

    def to_str(self, exclude_description: bool = False) -> str:
        lines = [f"Graph: {self.name}", "", "Node Labels:"]
        for e in self.entities:
            props = ", ".join(f"{k}: {v.value}" for k, v in e.properties.items())
            lines.append(f"  (:{e.label}) {{ {props} }}")
            if not exclude_description and e.description:
                lines.append(f"    Description: {e.description}")
        lines.append("")
        lines.append("Relationship Types:")
        for r in self.relations:
            props = ", ".join(f"{k}: {v.value}" for k, v in r.properties.items())
            prop_str = f" {{ {props} }}" if props else ""
            lines.append(f"  (:{r.subj_label})-[:{r.label}{prop_str}]->(:{r.obj_label})")
            if not exclude_description and r.description:
                lines.append(f"    Description: {r.description}")
        return "\n".join(lines)
