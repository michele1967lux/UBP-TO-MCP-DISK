"""
agentic_rag/prompts.py

Prompt templates for agentic RAG operations.

Includes:
- ReAct loop prompts
- Thought generation
- Action selection
- Observation processing
- Reflection prompts
- Answer synthesis

Multi-language support: EN/IT

v1.0.0: Initial release
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ============================================================================
# ReAct Loop Prompts
# ============================================================================


REACT_SYSTEM_EN = """You are an intelligent agent that answers questions by reasoning step-by-step and using tools.

Available tools:
{tools}

Follow this format for each step:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

After receiving an observation, continue with another Thought/Action or provide:
Thought: [Your final reasoning]
Final Answer: [Your comprehensive answer to the question]

Rules:
- Always think before acting
- Use tools when you need information
- Cite your sources when providing facts
- If you have enough information, provide the Final Answer
- Be concise but thorough"""


REACT_SYSTEM_IT = """Sei un agente intelligente che risponde alle domande ragionando passo-passo e usando strumenti.

Strumenti disponibili:
{tools}

Segui questo formato per ogni passo:
Thought: [Il tuo ragionamento su cosa fare dopo]
Action: [nome_strumento]
Action Input: {{"param1": "valore1", "param2": "valore2"}}

Dopo aver ricevuto un'osservazione, continua con un altro Thought/Action o fornisci:
Thought: [Il tuo ragionamento finale]
Final Answer: [La tua risposta completa alla domanda]

Regole:
- Pensa sempre prima di agire
- Usa gli strumenti quando hai bisogno di informazioni
- Cita le fonti quando fornisci fatti
- Se hai abbastanza informazioni, fornisci la Final Answer
- Sii conciso ma completo"""


REACT_STEP_EN = """Question: {query}

{history}

Now continue with your next Thought and Action (or Final Answer if you have enough information):"""


REACT_STEP_IT = """Domanda: {query}

{history}

Ora continua con il tuo prossimo Thought e Action (o Final Answer se hai abbastanza informazioni):"""


# ============================================================================
# Thought Generation
# ============================================================================


THOUGHT_PROMPT_EN = """Based on the current situation, what should be your next step?

Question: {query}

Information gathered so far:
{context}

Previous steps:
{history}

Think about:
1. What information do you still need?
2. Which tool would be most helpful?
3. Have you gathered enough to answer?

Provide your thought process:"""


THOUGHT_PROMPT_IT = """Basandoti sulla situazione attuale, quale dovrebbe essere il tuo prossimo passo?

Domanda: {query}

Informazioni raccolte finora:
{context}

Passi precedenti:
{history}

Pensa a:
1. Quali informazioni ti mancano ancora?
2. Quale strumento sarebbe più utile?
3. Hai raccolto abbastanza per rispondere?

Fornisci il tuo processo di pensiero:"""


# ============================================================================
# Action Selection
# ============================================================================


ACTION_SELECTION_EN = """Based on your thought, select the best action.

Thought: {thought}

Available tools:
{tools}

Select one action and its parameters in JSON format:
{{
    "tool": "<tool_name>",
    "parameters": {{
        "param1": "value1"
    }},
    "reason": "<why this tool>"
}}

If you don't need any tool and can answer directly, respond:
{{
    "tool": "none",
    "final_answer": "<your answer>"
}}"""


ACTION_SELECTION_IT = """Basandoti sul tuo pensiero, seleziona la migliore azione.

Thought: {thought}

Strumenti disponibili:
{tools}

Seleziona un'azione e i suoi parametri in formato JSON:
{{
    "tool": "<nome_strumento>",
    "parameters": {{
        "param1": "valore1"
    }},
    "reason": "<perché questo strumento>"
}}

Se non hai bisogno di strumenti e puoi rispondere direttamente, rispondi:
{{
    "tool": "none",
    "final_answer": "<la tua risposta>"
}}"""


# ============================================================================
# Observation Processing
# ============================================================================


OBSERVATION_PROMPT_EN = """You received the following observation from the tool.

Tool used: {tool_name}
Observation:
{observation}

Process this observation:
1. What relevant information did you find?
2. Does this answer the question fully or partially?
3. Do you need more information?

Provide your analysis:"""


OBSERVATION_PROMPT_IT = """Hai ricevuto la seguente osservazione dallo strumento.

Strumento usato: {tool_name}
Osservazione:
{observation}

Elabora questa osservazione:
1. Quali informazioni rilevanti hai trovato?
2. Questo risponde alla domanda completamente o parzialmente?
3. Hai bisogno di più informazioni?

Fornisci la tua analisi:"""


# ============================================================================
# Reflection Prompts
# ============================================================================


REFLECTION_PROMPT_EN = """Reflect on your progress so far.

Original question: {query}

Steps taken:
{history}

Information gathered:
{context}

Reflection questions:
1. Are you making progress toward answering the question?
2. Have you explored the right directions?
3. Is there a different approach you should try?
4. Do you have enough information to answer now?

Provide your reflection and decide next steps:"""


REFLECTION_PROMPT_IT = """Rifletti sui tuoi progressi finora.

Domanda originale: {query}

Passi compiuti:
{history}

Informazioni raccolte:
{context}

Domande di riflessione:
1. Stai facendo progressi verso la risposta?
2. Hai esplorato le direzioni giuste?
3. C'è un approccio diverso che dovresti provare?
4. Hai abbastanza informazioni per rispondere ora?

Fornisci la tua riflessione e decidi i prossimi passi:"""


# ============================================================================
# Answer Synthesis
# ============================================================================


SYNTHESIS_PROMPT_EN = """Synthesize a final answer based on all gathered information.

Question: {query}

Information sources:
{sources}

Reasoning trace:
{reasoning}

Instructions:
1. Combine all relevant information
2. Cite sources when stating facts
3. Be comprehensive but concise
4. Acknowledge any limitations or uncertainties
5. Structure your answer clearly

Provide your final answer:"""


SYNTHESIS_PROMPT_IT = """Sintetizza una risposta finale basata su tutte le informazioni raccolte.

Domanda: {query}

Fonti di informazione:
{sources}

Traccia di ragionamento:
{reasoning}

Istruzioni:
1. Combina tutte le informazioni rilevanti
2. Cita le fonti quando affermi fatti
3. Sii completo ma conciso
4. Riconosci eventuali limitazioni o incertezze
5. Struttura la risposta chiaramente

Fornisci la tua risposta finale:"""


# ============================================================================
# Error Recovery
# ============================================================================


ERROR_RECOVERY_EN = """The previous action failed. Decide how to proceed.

Original question: {query}
Failed action: {failed_action}
Error: {error}

Options:
1. Retry with different parameters
2. Use a different tool
3. Answer with available information
4. Acknowledge the limitation

What should you do?"""


ERROR_RECOVERY_IT = """L'azione precedente è fallita. Decidi come procedere.

Domanda originale: {query}
Azione fallita: {failed_action}
Errore: {error}

Opzioni:
1. Riprova con parametri diversi
2. Usa uno strumento diverso
3. Rispondi con le informazioni disponibili
4. Riconosci la limitazione

Cosa dovresti fare?"""


# ============================================================================
# Parallel Planning
# ============================================================================


PARALLEL_PLANNING_EN = """Plan how to gather information efficiently for this query.

Query: {query}

Available tools:
{tools}

Instructions:
1. Identify what information is needed
2. Determine which queries can run IN PARALLEL (no dependencies)
3. Identify sequential dependencies
4. Optimize for speed while ensuring completeness

Respond in JSON:
{{
    "parallel_actions": [
        {{"tool": "...", "params": {{}}, "purpose": "..."}}
    ],
    "sequential_actions": [
        {{"tool": "...", "params": {{}}, "depends_on": [...], "purpose": "..."}}
    ],
    "reasoning": "..."
}}"""


PARALLEL_PLANNING_IT = """Pianifica come raccogliere informazioni in modo efficiente per questa query.

Query: {query}

Strumenti disponibili:
{tools}

Istruzioni:
1. Identifica quali informazioni sono necessarie
2. Determina quali query possono essere eseguite IN PARALLELO (senza dipendenze)
3. Identifica le dipendenze sequenziali
4. Ottimizza per velocità garantendo completezza

Rispondi in JSON:
{{
    "parallel_actions": [
        {{"tool": "...", "params": {{}}, "purpose": "..."}}
    ],
    "sequential_actions": [
        {{"tool": "...", "params": {{}}, "depends_on": [...], "purpose": "..."}}
    ],
    "reasoning": "..."
}}"""


# ============================================================================
# Template Registry
# ============================================================================


TEMPLATES = {
    "en": {
        "react_system": REACT_SYSTEM_EN,
        "react_step": REACT_STEP_EN,
        "thought": THOUGHT_PROMPT_EN,
        "action_selection": ACTION_SELECTION_EN,
        "observation": OBSERVATION_PROMPT_EN,
        "reflection": REFLECTION_PROMPT_EN,
        "synthesis": SYNTHESIS_PROMPT_EN,
        "error_recovery": ERROR_RECOVERY_EN,
        "parallel_planning": PARALLEL_PLANNING_EN,
    },
    "it": {
        "react_system": REACT_SYSTEM_IT,
        "react_step": REACT_STEP_IT,
        "thought": THOUGHT_PROMPT_IT,
        "action_selection": ACTION_SELECTION_IT,
        "observation": OBSERVATION_PROMPT_IT,
        "reflection": REFLECTION_PROMPT_IT,
        "synthesis": SYNTHESIS_PROMPT_IT,
        "error_recovery": ERROR_RECOVERY_IT,
        "parallel_planning": PARALLEL_PLANNING_IT,
    },
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_template(template_name: str, language: str = "en") -> str:
    """Get a template by name and language."""
    lang_templates = TEMPLATES.get(language, TEMPLATES["en"])
    return lang_templates.get(template_name, TEMPLATES["en"].get(template_name, ""))


def detect_language(text: str) -> str:
    """Simple language detection."""
    italian_markers = {
        "come", "cosa", "perché", "quando", "dove", "chi", "quale",
        "il", "la", "lo", "gli", "le", "un", "una", "uno",
        "è", "sono", "essere", "avere", "fare", "per", "con", "di",
    }
    words = set(text.lower().split())
    italian_ratio = len(words & italian_markers) / max(len(words), 1)
    return "it" if italian_ratio > 0.15 else "en"


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """Format tools list for inclusion in prompts."""
    lines = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", {}).get("properties", {})
        
        param_str = ", ".join([
            f"{p}: {params[p].get('type', 'any')}"
            for p in params
        ])
        
        lines.append(f"- {name}({param_str}): {desc}")
    
    return "\n".join(lines)


def format_history_for_prompt(steps: List[Dict[str, Any]], max_steps: int = 10) -> str:
    """Format step history for inclusion in prompts."""
    lines = []
    
    for step in steps[-max_steps:]:
        step_type = step.get("type", "unknown")
        content = step.get("content", "")
        
        if step_type == "thought":
            lines.append(f"Thought: {content}")
        elif step_type == "action":
            tool = step.get("tool", "unknown")
            params = step.get("params", {})
            lines.append(f"Action: {tool}")
            lines.append(f"Action Input: {params}")
        elif step_type == "observation":
            obs = content[:500] if len(content) > 500 else content
            lines.append(f"Observation: {obs}")
    
    return "\n".join(lines)


def format_sources_for_synthesis(sources: List[Dict[str, Any]]) -> str:
    """Format sources for synthesis prompt."""
    lines = []
    
    for i, source in enumerate(sources, 1):
        content = source.get("content", source.get("text", str(source)))
        if len(content) > 300:
            content = content[:300] + "..."
        
        tool = source.get("tool", source.get("source", "unknown"))
        lines.append(f"[Source {i} - {tool}]: {content}")
    
    return "\n\n".join(lines)


def parse_react_response(response: str) -> Dict[str, Any]:
    """Parse a ReAct-style response."""
    result = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
    }
    
    # Extract Thought
    thought_match = response.split("Thought:")
    if len(thought_match) > 1:
        thought_text = thought_match[-1].split("Action:")[0].strip()
        result["thought"] = thought_text
    
    # Extract Final Answer
    if "Final Answer:" in response:
        answer_match = response.split("Final Answer:")[-1].strip()
        result["final_answer"] = answer_match
        return result
    
    # Extract Action
    if "Action:" in response:
        action_text = response.split("Action:")[-1]
        action_name = action_text.split("\n")[0].strip()
        if "Action Input:" in action_name:
            action_name = action_name.split("Action Input:")[0].strip()
        result["action"] = action_name
    
    # Extract Action Input
    if "Action Input:" in response:
        import json
        input_text = response.split("Action Input:")[-1].strip()
        # Try to find JSON
        try:
            # Find JSON object
            start = input_text.find("{")
            end = input_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = input_text[start:end]
                result["action_input"] = json.loads(json_str)
        except json.JSONDecodeError:
            result["action_input"] = {"query": input_text.split("\n")[0]}
    
    return result
