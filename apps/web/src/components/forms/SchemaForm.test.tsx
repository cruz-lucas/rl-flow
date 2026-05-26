import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SchemaForm } from "./SchemaForm";

describe("SchemaForm", () => {
  it("preserves numeric enum values", () => {
    const onChange = vi.fn();
    render(
      <SchemaForm
        schema={{ type: "object", properties: { size: { type: "integer", enum: [5, 6, 8, 16] } } }}
        value={{ size: 5 }}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByDisplayValue("5"), { target: { value: "3" } });

    expect(onChange).toHaveBeenCalledWith({ size: 16 });
  });

  it("parses nullable integers as numbers", () => {
    const onChange = vi.fn();
    render(
      <SchemaForm
        schema={{ type: "object", properties: { max_steps: { type: ["integer", "null"], minimum: 1 } } }}
        value={{ max_steps: null }}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("max_steps"), { target: { value: "2048" } });

    expect(onChange).toHaveBeenCalledWith({ max_steps: 2048 });
  });

  it("hides inspector-hidden schema fields", () => {
    render(
      <SchemaForm
        schema={{
          type: "object",
          properties: {
            epsilon_start: { type: "number" },
            eps_start: { type: "number", "x-inspector-hidden": true },
          },
        }}
        value={{ epsilon_start: 1.0, eps_start: 1.0 }}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByLabelText("epsilon_start")).toBeInTheDocument();
    expect(screen.queryByLabelText("eps_start")).not.toBeInTheDocument();
  });
});
