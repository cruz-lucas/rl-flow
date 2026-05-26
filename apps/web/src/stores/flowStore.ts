import { create } from "zustand";
import type { Edge, Node } from "reactflow";
import type { ComponentSpec, WorkflowSpec } from "../types/schema";

export interface FlowNodeData {
  label: string;
  component: ComponentSpec;
  config: Record<string, unknown>;
}

interface FlowState {
  nodes: Node<FlowNodeData>[];
  edges: Edge[];
  selectedNodeId?: string;
  backend: "local" | "slurm";
  workflowName: string;
  setNodes: (nodes: Node<FlowNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  setSelectedNodeId: (id?: string) => void;
  setBackend: (backend: "local" | "slurm") => void;
  setWorkflowName: (name: string) => void;
  addComponent: (component: ComponentSpec, position: { x: number; y: number }) => void;
  updateNodeConfig: (nodeId: string, config: Record<string, unknown>) => void;
  loadWorkflow: (workflow: WorkflowSpec, components: ComponentSpec[]) => void;
  toWorkflow: () => WorkflowSpec;
}

export const useFlowStore = create<FlowState>((set, get) => ({
  nodes: [],
  edges: [],
  backend: "local",
  workflowName: "untitled_workflow",
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),
  setSelectedNodeId: (selectedNodeId) => set({ selectedNodeId }),
  setBackend: (backend) => set({ backend }),
  setWorkflowName: (workflowName) => set({ workflowName }),
  addComponent: (component, position) =>
    set((state) => {
      const id = `${component.kind}-${state.nodes.length + 1}`;
      return {
        nodes: [
          ...state.nodes,
          {
            id,
            type: "componentNode",
            position,
            data: { label: component.display_name, component, config: { ...component.defaults } },
          },
        ],
        selectedNodeId: id,
      };
    }),
  updateNodeConfig: (nodeId, config) =>
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, config } } : node,
      ),
    })),
  loadWorkflow: (workflow, components) =>
    set({
      workflowName: workflow.name,
      backend: workflow.execution.backend,
      nodes: workflow.nodes.map((node) => {
        const component = components.find((item) => item.id === node.component);
        if (!component) {
          throw new Error(`Missing component ${node.component}`);
        }
        return {
          id: node.id,
          type: "componentNode",
          position: node.position,
          data: { label: component.display_name, component, config: { ...component.defaults, ...node.config } },
        };
      }),
      edges: workflow.edges.map((edge, index) => ({
        id: `edge-${index}`,
        source: edge.from_node,
        sourceHandle: edge.from_port,
        target: edge.to_node,
        targetHandle: edge.to_port,
      })),
      selectedNodeId: undefined,
    }),
  toWorkflow: () => {
    const state = get();
    return {
      name: state.workflowName,
      description: "",
      metadata: {},
      execution: { backend: state.backend, options: {} },
      nodes: state.nodes.map((node) => ({
        id: node.id,
        component: node.data.component.id,
        config: node.data.config,
        position: { x: node.position.x, y: node.position.y },
      })),
      edges: state.edges.map((edge) => ({
        from_node: edge.source,
        from_port: edge.sourceHandle ?? "out",
        to_node: edge.target,
        to_port: edge.targetHandle ?? "in",
      })),
    };
  },
}));
