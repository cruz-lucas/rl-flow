import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { App } from "./App";

describe("App", () => {
  beforeAll(() => {
    class ResizeObserver {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = vi.fn();
    }
    vi.stubGlobal("ResizeObserver", ResizeObserver);
  });

  it("renders navigation", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(screen.getByText("rl-flow")).toBeInTheDocument();
    expect(screen.getByText("Flow")).toBeInTheDocument();
    expect(screen.getByText("Environment")).toBeInTheDocument();
    expect(screen.getByText("Dataset")).toBeInTheDocument();
    expect(screen.getByText("Offline RL")).toBeInTheDocument();
    expect(screen.queryByText("Load CartPole")).not.toBeInTheDocument();
  });
});
