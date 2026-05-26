import { Settings } from "lucide-react";
import { SchemaForm } from "../forms/SchemaForm";
import { useFlowStore } from "../../stores/flowStore";

export function InspectorPanel() {
  const { nodes, selectedNodeId, updateNodeConfig } = useFlowStore();
  const node = nodes.find((item) => item.id === selectedNodeId);
  return (
    <aside className="inspector">
      <div className="panel-title">
        <Settings size={16} />
        Inspector
      </div>
      {!node ? (
        <div className="empty-state">Select a node</div>
      ) : (
        <>
          <div className="selected-heading">
            <strong>{node.data.label}</strong>
            <span>{node.data.component.id}</span>
          </div>
          <SchemaForm
            schema={node.data.component.config_schema}
            value={node.data.config}
            onChange={(config) => updateNodeConfig(node.id, config)}
          />
        </>
      )}
    </aside>
  );
}
