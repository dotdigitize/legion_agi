
# Legion AI
![LegionAI1 dotdigitze creativedisruptor](https://github.com/user-attachments/assets/8576e96a-84ed-47ac-89f3-9bde269390b6)

Welcome to **Legion AI** - a multi-agent system that spawns intelligent agents capable of solving complex problems collaboratively. This README will guide you through the setup, features, and future applications of the Legion AI system.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
  - [PAST Method](#past-method)
  - [RAFT Method](#raft-method)
  - [EAT Method](#eat-method)
- [Example Usage](#example-usage)
- [Applications](#applications)
- [Installation](#installation)
- [Future Scope](#future-scope)
- [Collaboration](#collaboration)

---

## Overview

**Legion AI** is designed to dynamically spawn agents based on user input, allowing these agents to collaborate, critique, and refine ideas to provide sophisticated solutions to complex tasks. Using state-of-the-art methods such as PAST, RAFT, and EAT, it enables dynamic reasoning that evolves over time, enhancing the overall thought process.

---

## Features

- **Multi-Agent Spawning:** Dynamically generates multiple expert agents based on user input.
- **Collaborative Reasoning:** Agents work together to provide insights, critique solutions, and refine ideas.
- **Advanced Methods:** Utilizes methods like PAST, RAFT, and EAT for complex problem-solving.
- **Guided Conversations:** A guiding agent ensures that the conversation stays focused on the user’s original question.

---

## Methodology

### PAST Method

The **PAST** (Personas, Actions, Solutions, and Task) method is the initial stage of reasoning where the system spawns the appropriate experts (personas) based on the user's query. Each agent is assigned a specific action to perform, contributing to an overall task.

### RAFT Method

The **RAFT** (Reasoning, Analysis, Feedback, and Thought) method comes into play in the second phase of reasoning, where agents exchange feedback and critique one another's thoughts. This enables deeper reasoning and leads to more refined solutions.

### EAT Method

**EAT** (Evaluation, Action, and Testing) is the final iterative phase. Agents test the viability of their suggestions by critically evaluating the effectiveness of their proposals and providing actionable recommendations to the user.

---

## Example Usage

Here is a basic example of how to interact with the Legion AI system:

```python
# Import the necessary modules
from legion_ai import ChatManager

# Start a chat session
chat_manager = ChatManager()
chat_manager.start_chat()
```

During the interaction, you can ask complex questions like:

```
How can I optimize my deep learning model for energy efficiency?
```

The system will then spawn agents like Geoffrey Hinton, Claude Shannon, and Ada Lovelace to collaborate on the solution.

---

## Applications

Legion AI can be applied in various fields:
- **Artificial General Intelligence (AGI):** Simulating the reasoning processes behind AGI.
- **Collaborative Problem Solving:** Useful for brainstorming sessions, research, and decision-making.
- **Machine Learning & Deep Learning:** Optimize models and processes with expert knowledge.
- **Business and Strategy:** Spawn experts in economics, strategy, and innovation to refine business decisions.

---

## Installation

To set up Legion AI, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/dotdigitize/legion-ai.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd legion-ai
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the system:
   ```bash
   python legion_ai.py
   ```

---

## Future Scope

Legion AI is a continuously evolving project. Future plans include:
- **Agent Memory:** Allowing agents to remember previous conversations and build on past insights.
- **Cross-Domain Knowledge Integration:** Expanding the knowledge base to include more specialized fields.
- **Neural Network Integration:** Implementing real-time learning for agents to evolve based on feedback.

---

## Collaboration

We welcome contributions from the community! If you have ideas, suggestions, or improvements, feel free to collaborate with us.

### How to contribute:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request with a detailed description of your changes.

Stay connected with the project and collaborate on new ideas!

---

**Contact:** For inquiries or collaborations, reach out to the project maintainers at [email@example.com](mailto:email@example.com).

Enjoy using **Legion AI**!
