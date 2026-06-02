import fs from "node:fs";
import path from "node:path";

const BOOTSTRAP_HEADER = "## CitationForce Constraint";
const CONDITION_MARKER_FILE = ".groundingbench_citation_force_condition";
const TOOL_CONDITIONS = new Set(["C2", "C3", "Cm"]);

function readConditionMarker(workspaceDir: string): string | null {
  const markerPath = path.join(workspaceDir, CONDITION_MARKER_FILE);
  if (!fs.existsSync(markerPath)) {
    return null;
  }
  const value = fs.readFileSync(markerPath, "utf8").trim();
  return value || null;
}

function bootstrapHasCitationForce(workspaceDir: string): boolean {
  const bootstrapPath = path.join(workspaceDir, "BOOTSTRAP.md");
  if (!fs.existsSync(bootstrapPath)) {
    return false;
  }
  return fs.readFileSync(bootstrapPath, "utf8").includes(BOOTSTRAP_HEADER);
}

function iterMemoryFiles(workspaceDir: string): string[] {
  const files: string[] = [];
  const memoryMd = path.join(workspaceDir, "MEMORY.md");
  if (fs.existsSync(memoryMd)) {
    files.push(memoryMd);
  }

  const memoryDir = path.join(workspaceDir, "memory");
  if (fs.existsSync(memoryDir)) {
    for (const entry of fs.readdirSync(memoryDir).sort().reverse()) {
      if (entry.endsWith(".md")) {
        files.push(path.join(memoryDir, entry));
      }
    }
  }

  for (const entry of fs.readdirSync(workspaceDir).sort()) {
    if (!entry.endsWith(".md") || entry === "MEMORY.md") {
      continue;
    }
    files.push(path.join(workspaceDir, entry));
  }

  return files;
}

function createArtifactRecallTool(workspaceDir: string) {
  return {
    name: "artifact_recall",
    label: "Artifact Recall",
    description:
      "Retrieve a specific artifact from workspace memory and return source paths for citation-aware grounding.",
    promptSnippet:
      "artifact_recall: retrieve a specific artifact from workspace memory and return source_paths for citation.",
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        artifact_id: {
          type: "string",
          description: "Target artifact id or exact file/path fragment, e.g. ss_abc123 or cqa_deadbeef.",
        },
        keywords: {
          type: "array",
          items: { type: "string" },
          description: "Optional extra keywords to search for inside memory files.",
        },
      },
    },
    async execute(_toolCallId: string, rawParams: unknown) {
      const params = (rawParams ?? {}) as { artifact_id?: string; keywords?: string[] };
      const searchTerms = [
        ...(params.artifact_id ? [params.artifact_id] : []),
        ...((params.keywords ?? []).filter(Boolean)),
      ];

      if (searchTerms.length === 0) {
        return {
          content: [{ type: "text", text: "" }],
          details: { text: "", source_paths: [], artifact_id: params.artifact_id ?? "", found: false },
        };
      }

      const matches: Array<{ path: string; snippet: string }> = [];
      for (const absPath of iterMemoryFiles(workspaceDir)) {
        const content = fs.readFileSync(absPath, "utf8");
        for (const term of searchTerms) {
          if (!term || !content.includes(term)) {
            continue;
          }
          const idx = content.indexOf(term);
          const snippet = content.slice(Math.max(0, idx - 200), idx + 500).trim();
          matches.push({
            path: path.relative(workspaceDir, absPath) || path.basename(absPath),
            snippet,
          });
          break;
        }
      }

      if (matches.length === 0) {
        return {
          content: [{ type: "text", text: "" }],
          details: { text: "", source_paths: [], artifact_id: params.artifact_id ?? "", found: false },
        };
      }

      const text = matches
        .map((m) => `[Source: ${m.path}]\n${m.snippet}`)
        .join("\n\n---\n\n");
      return {
        content: [{ type: "text", text }],
        details: {
          text,
          source_paths: matches.map((m) => m.path),
          artifact_id: params.artifact_id ?? "",
          found: true,
          results: matches.map((m) => ({ path: m.path, snippet: m.snippet })),
        },
      };
    },
  };
}

const plugin = {
  id: "groundingbench-artifact-recall",
  name: "GroundingBench Artifact Recall",
  description: "Condition-gated artifact_recall tool for CitationForce C2/C3/Cm trials.",
  register(api: any) {
    api.registerTool(
      (ctx: { workspaceDir?: string }) => {
        const workspaceDir = ctx.workspaceDir;
        if (!workspaceDir || !fs.existsSync(workspaceDir)) {
          return null;
        }
        const cond = readConditionMarker(workspaceDir);
        if (!cond || !TOOL_CONDITIONS.has(cond)) {
          return null;
        }
        if (!bootstrapHasCitationForce(workspaceDir)) {
          return null;
        }
        return createArtifactRecallTool(workspaceDir);
      },
      { name: "artifact_recall" },
    );
  },
};

export default plugin;
