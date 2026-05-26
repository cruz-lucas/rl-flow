import type { JsonSchema } from "../../types/schema";

interface SchemaFormProps {
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
}

export function SchemaForm({ schema, value, onChange }: SchemaFormProps) {
  const properties = schema.properties ?? {};
  const entries = Object.entries(properties).filter(([, field]) => !field["x-inspector-hidden"]);
  if (entries.length === 0) {
    return <div className="empty-state">No configuration fields</div>;
  }
  return (
    <form className="schema-form">
      {entries.map(([key, field]) => (
        <label key={key} className="field">
          <span>{key}</span>
          <FieldInput
            schema={field}
            value={value[key] ?? field.default ?? ""}
            onChange={(nextValue) => onChange({ ...value, [key]: nextValue })}
          />
        </label>
      ))}
    </form>
  );
}

function FieldInput({
  schema,
  value,
  onChange,
}: {
  schema: JsonSchema;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (schema.enum) {
    const selectedIndex = Math.max(0, schema.enum.findIndex((option) => valuesEqual(option, value)));
    return (
      <select
        value={String(selectedIndex)}
        onChange={(event) => onChange(schema.enum?.[Number(event.target.value)] ?? "")}
      >
        {schema.enum.map((option, index) => (
          <option key={`${String(option)}-${index}`} value={String(index)}>
            {String(option)}
          </option>
        ))}
      </select>
    );
  }
  if (hasSchemaType(schema, "boolean")) {
    return (
      <input
        type="checkbox"
        checked={Boolean(value)}
        onChange={(event) => onChange(event.target.checked)}
      />
    );
  }
  if (hasSchemaType(schema, "integer") || hasSchemaType(schema, "number")) {
    const numericType = hasSchemaType(schema, "integer") ? "integer" : "number";
    const inputValue = value === null || value === undefined || value === "" ? "" : Number(value);
    return (
      <input
        type="number"
        value={inputValue}
        min={schema.minimum}
        max={schema.maximum}
        step={numericType === "integer" ? 1 : "any"}
        onChange={(event) => {
          if (event.target.value === "" && hasSchemaType(schema, "null")) {
            onChange(null);
            return;
          }
          const parsed = numericType === "integer" ? Number.parseInt(event.target.value, 10) : Number(event.target.value);
          onChange(Number.isNaN(parsed) ? 0 : parsed);
        }}
      />
    );
  }
  if (hasSchemaType(schema, "array")) {
    return (
      <input
        value={JSON.stringify(value ?? [])}
        onChange={(event) => {
          try {
            const parsed = JSON.parse(event.target.value);
            onChange(Array.isArray(parsed) ? parsed : []);
          } catch {
            onChange(event.target.value);
          }
        }}
      />
    );
  }
  return <input value={String(value)} onChange={(event) => onChange(event.target.value)} />;
}

function hasSchemaType(schema: JsonSchema, type: string): boolean {
  if (Array.isArray(schema.type)) {
    return schema.type.includes(type);
  }
  return schema.type === type;
}

function valuesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}
