import { useMemo, useRef } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
} from "reactflow";
import "reactflow/dist/style.css";
import { ComponentNode } from "./ComponentNode";
import { useFlowStore } from "../../stores/flowStore";
import type { ComponentSpec } from "../../types/schema";

const nodeTypes: NodeTypes = { componentNode: ComponentNode };

export function WorkflowCanvas({ components }: { components: ComponentSpec[] }) {
  return (
    <ReactFlowProvider>
      <CanvasInner components={components} />
    </ReactFlowProvider>
  );
}

function CanvasInner({ components }: { components: ComponentSpec[] }) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const reactFlow = useReactFlow();
  const { nodes, edges, setNodes, setEdges, setSelectedNodeId, addComponent } = useFlowStore();
  const componentById = useMemo(() => new Map(components.map((component) => [component.id, component])), [components]);

  const onConnect = (connection: Connection) => {
    setEdges(addEdge({ ...connection, id: `${connection.source}-${connection.sourceHandle}-${connection.target}-${connection.targetHandle}` }, edges));
  };

  return (
    <div
      className="canvas"
      ref={wrapperRef}
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
      onDrop={(event) => {
        event.preventDefault();
        const componentId = event.dataTransfer.getData("application/rlflow-component");
        const component = componentById.get(componentId);
        if (!component || !wrapperRef.current) return;
        const bounds = wrapperRef.current.getBoundingClientRect();
        const position = reactFlow.project({ x: event.clientX - bounds.left, y: event.clientY - bounds.top });
        addComponent(component, position);
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={(changes) => setNodes(applyNodeChanges(changes, nodes) as Node[])}
        onEdgesChange={(changes) => setEdges(applyEdgeChanges(changes, edges) as Edge[])}
        onConnect={onConnect}
        onNodeClick={(_, node) => setSelectedNodeId(node.id)}
        fitView
      >
        <Background gap={18} size={1} />
        <MiniMap pannable zoomable />
        <Controls />
      </ReactFlow>
    </div>
  );
}
