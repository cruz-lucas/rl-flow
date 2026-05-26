import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import type { ComponentSpec } from "../../types/schema";

interface ComponentPaletteProps {
  components: ComponentSpec[];
  isLoading?: boolean;
  error?: Error | null;
}

export function ComponentPalette({ components, isLoading, error }: ComponentPaletteProps) {
  const [collapsedSources, setCollapsedSources] = useState<Record<string, boolean>>({});
  const groups = components.reduce<Record<string, Record<string, ComponentSpec[]>>>((acc, component) => {
    const source = component.source ?? "custom";
    acc[source] = acc[source] ?? {};
    acc[source][component.kind] = [...(acc[source][component.kind] ?? []), component];
    return acc;
  }, {});
  const groupEntries = Object.entries(groups).sort(([left], [right]) => left.localeCompare(right));

  const toggleSource = (source: string) => {
    setCollapsedSources((current) => ({ ...current, [source]: !current[source] }));
  };

  return (
    <aside className="palette">
      <div className="panel-title">Components</div>
      {isLoading && <div className="empty-state">Loading components</div>}
      {error && <div className="error-state">Could not load components from the API</div>}
      {!isLoading && !error && components.length === 0 && <div className="empty-state">No components registered</div>}
      {groupEntries.map(([source, kindGroups]) => {
        const isCollapsed = collapsedSources[source] ?? false;
        const sourceCount = Object.values(kindGroups).reduce((count, items) => count + items.length, 0);
        return (
          <section key={source} className="palette-group">
            <button
              type="button"
              className="source-toggle"
              aria-expanded={!isCollapsed}
              onClick={() => toggleSource(source)}
            >
              <span className="source-title">
                {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                {source}
              </span>
              <span className="source-count">{sourceCount}</span>
            </button>
            {!isCollapsed &&
              Object.entries(kindGroups)
                .sort(([left], [right]) => left.localeCompare(right))
                .map(([kind, items]) => (
                  <div key={`${source}-${kind}`}>
                    <div className="group-subtitle">{kind}</div>
                    {items.map((component) => (
                      <button
                        key={component.id}
                        type="button"
                        className="palette-item"
                        draggable
                        onDragStart={(event) => {
                          event.dataTransfer.setData("application/rlflow-component", component.id);
                          event.dataTransfer.effectAllowed = "copy";
                        }}
                      >
                        <span>{component.display_name}</span>
                        <small>{component.id}</small>
                      </button>
                    ))}
                  </div>
                ))}
        </section>
        );
      })}
    </aside>
  );
}
