import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";
import type { FlowNodeData } from "../../stores/flowStore";

export function ComponentNode({ data, selected }: NodeProps<FlowNodeData>) {
  return (
    <div className={`flow-node ${selected ? "selected" : ""}`}>
      <div className="node-kind">{data.component.kind}</div>
      <div className="node-title">{data.label}</div>
      <div className="port-list input-ports">
        {data.component.input_ports.map((port, index) => (
          <div key={port.name} className="port-row">
            <Handle
              type="target"
              id={port.name}
              position={Position.Left}
              style={{ top: 56 + index * 22 }}
            />
            <span>{port.name}</span>
          </div>
        ))}
      </div>
      <div className="port-list output-ports">
        {data.component.output_ports.map((port, index) => (
          <div key={port.name} className="port-row right">
            <span>{port.name}</span>
            <Handle
              type="source"
              id={port.name}
              position={Position.Right}
              style={{ top: 56 + index * 22 }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
