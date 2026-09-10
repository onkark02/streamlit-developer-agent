"""Streamlit UI for the developer code-review agent."""

import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree, tracing_context
import streamlit as st

SYSTEM_PROMPT = (
    "You are a senior developer performing a code review. Use "
    "static_code_check on the snippet, then write a short, "
    "constructive review comment covering what to fix and why."
)

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "o4-mini",
    "o3-mini",
]

DEFAULT_LANGSMITH_PROJECT = "developer-review-agent"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


@tool
def static_code_check(code: str) -> str:
    """Run a lightweight static check on a Python code snippet."""
    issues = []
    if '"""' not in code and "'''" not in code:
        issues.append("No docstring found.")
    if "TODO" in code:
        issues.append("Contains TODO comment(s) left in the code.")
    if code.count("\n") > 40:
        issues.append("Function/file may be too long — consider splitting it.")
    return "; ".join(issues) if issues else "No obvious issues found."


def build_agent(api_key: str, model: str):
    llm = ChatOpenAI(model=model, api_key=api_key)
    return create_agent(llm, [static_code_check], system_prompt=SYSTEM_PROMPT)


def configure_langsmith(
    *,
    enabled: bool,
    api_key: str,
    project: str,
    endpoint: str,
) -> Client | None:
    """Enable or disable LangSmith tracing for this Streamlit session."""
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("[langsmith] tracing disabled")
        return None

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    client = Client(api_key=api_key, api_url=endpoint or None)
    print(f"[langsmith] tracing enabled project={project} endpoint={endpoint or 'default'}")
    return client


def run_url_from_tree() -> str | None:
    run = get_current_run_tree()
    if run is None:
        return None
    try:
        return run.get_url()
    except Exception as exc:
        print(f"[langsmith] could not resolve run URL: {exc}")
        return None


def collect_sources(snippet: str, uploaded_files) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    if snippet.strip():
        sources.append(("pasted snippet", snippet))
    for uploaded in uploaded_files or []:
        text = uploaded.read().decode("utf-8", errors="replace")
        sources.append((uploaded.name, text))
    return sources


def extract_output(result: dict) -> str:
    messages = result.get("messages") or []
    if not messages:
        return "No review was returned."
    last = messages[-1]
    content = getattr(last, "content", last)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts).strip() or "No review was returned."
    return str(content).strip() or "No review was returned."


def extract_tool_steps(result: dict) -> list[tuple[str, str, str]]:
    steps: list[tuple[str, str, str]] = []
    for message in result.get("messages") or []:
        name = getattr(message, "type", "") or message.__class__.__name__
        if name in {"tool", "ToolMessage"}:
            tool_name = getattr(message, "name", "tool")
            content = getattr(message, "content", "")
            steps.append((tool_name, str(content), "observation"))
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if isinstance(call, dict):
                steps.append(
                    (
                        str(call.get("name", "tool")),
                        str(call.get("args", "")),
                        "call",
                    )
                )
    return steps


def review_source(
    agent,
    name: str,
    code: str,
    *,
    model: str,
    langsmith_client: Client | None,
    langsmith_project: str,
) -> tuple[dict, str | None]:
    print(f"[review] starting review for: {name}")
    print(f"[review] model={model} code length: {len(code)} chars, lines: {code.count(chr(10)) + 1}")
    invoke_kwargs = {
        "messages": [
            {
                "role": "user",
                "content": f"Review this code from {name}:\n{code}",
            }
        ]
    }
    if langsmith_client is None:
        result = agent.invoke(invoke_kwargs)
        print(f"[review] finished review for: {name} (tracing off)")
        return result, None

    with tracing_context(
        project_name=langsmith_project,
        enabled=True,
        client=langsmith_client,
        tags=["developer-review-agent", "streamlit", model],
        metadata={"source": name, "model": model},
    ):
        result = agent.invoke(invoke_kwargs)
        run_url = run_url_from_tree()
    print(f"[review] finished review for: {name} langsmith_url={run_url}")
    return result, run_url


def main() -> None:
    st.set_page_config(page_title="Developer Review Agent", page_icon="🧑‍💻")
    st.title("Developer Review Agent")
    st.caption(
        "Paste a Python snippet and/or upload `.py` files. "
        "The agent runs `static_code_check`, then writes a short review."
    )

    with st.sidebar:
        st.header("OpenAI")
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only for this session. It is not written to disk.",
        )
        model = st.selectbox(
            "Model",
            MODEL_OPTIONS,
            index=0,
            help="Chat model used by the review agent.",
        )

        st.header("LangSmith")
        enable_tracing = st.checkbox(
            "Enable LangSmith tracing",
            value=True,
            help="Send agent, LLM, and tool runs to LangSmith for traceability.",
        )
        langsmith_api_key = st.text_input(
            "LangSmith API key",
            type="password",
            help="From LangSmith Settings → API Keys.",
            disabled=not enable_tracing,
        )
        langsmith_project = st.text_input(
            "LangSmith project",
            value=DEFAULT_LANGSMITH_PROJECT,
            help="Traces are grouped under this project name.",
            disabled=not enable_tracing,
        )
        langsmith_endpoint = st.text_input(
            "LangSmith endpoint",
            value=DEFAULT_LANGSMITH_ENDPOINT,
            help="Default is LangSmith Cloud. Change this for self-hosted.",
            disabled=not enable_tracing,
        )
        st.caption("Traces include the model, source name, and tool calls.")

    snippet = st.text_area(
        "Python snippet",
        height=220,
        placeholder="def process(data):\n    ...",
    )
    uploaded_files = st.file_uploader(
        "Python files to review",
        type=["py"],
        accept_multiple_files=True,
    )

    if not st.button("Run review", type="primary"):
        return

    if not api_key.strip():
        st.error("Enter an OpenAI API key in the sidebar.")
        return

    if enable_tracing and not langsmith_api_key.strip():
        st.error("Enter a LangSmith API key, or disable tracing in the sidebar.")
        return

    sources = collect_sources(snippet, uploaded_files)
    if not sources:
        st.error("Paste a snippet or upload at least one `.py` file.")
        return

    langsmith_client = configure_langsmith(
        enabled=enable_tracing,
        api_key=langsmith_api_key.strip(),
        project=langsmith_project.strip() or DEFAULT_LANGSMITH_PROJECT,
        endpoint=langsmith_endpoint.strip(),
    )
    agent = build_agent(api_key.strip(), model)
    for name, code in sources:
        st.subheader(name)
        with st.expander("Code submitted", expanded=False):
            st.code(code, language="python")

        with st.spinner(f"Reviewing {name} with {model}..."):
            try:
                result, run_url = review_source(
                    agent,
                    name,
                    code,
                    model=model,
                    langsmith_client=langsmith_client,
                    langsmith_project=langsmith_project.strip()
                    or DEFAULT_LANGSMITH_PROJECT,
                )
            except Exception as vis:
                st.error(f"Review failed for {name}: {vis}")
                continue

        st.markdown(extract_output(result))
        if run_url:
            st.link_button("Open LangSmith trace", run_url)
        elif enable_tracing:
            st.caption(
                f"Trace sent to LangSmith project `{langsmith_project.strip() or DEFAULT_LANGSMITH_PROJECT}`."
            )
        steps = extract_tool_steps(result)
        if steps:
            with st.expander("Agent steps (tool calls)", expanded=False):
                for index, (tool_name, detail, kind) in enumerate(steps, start=1):
                    st.markdown(f"**Step {index}: `{tool_name}` ({kind})**")
                    st.write(detail)


if __name__ == "__main__":
    main()
