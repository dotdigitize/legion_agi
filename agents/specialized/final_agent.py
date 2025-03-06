"""
Final Agent for Legion AGI System

This specialized agent focuses on compiling and integrating results from other agents
into a comprehensive final solution. It takes contributions from multiple agents and
creates a coherent, structured response with proper organization and formatting.
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

import ollama
import re
import json

from legion_agi.config import DEFAULT_MODEL
from legion_agi.utils.logger import LogContext, PerformanceTimer


class FinalAgent:
    """
    Final agent for compiling integrated solutions.
    Takes inputs from multiple specialized agents and produces a cohesive final output.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize Final Agent.
        
        Args:
            model: LLM model to use for compilation
        """
        self.model = model
        self.compilation_history = []
        self.formatting_templates = self._load_formatting_templates()
        
        logger.info(f"Final Agent initialized with model {model}")
        
    def _load_formatting_templates(self) -> Dict[str, str]:
        """
        Load formatting templates for different output types.
        
        Returns:
            Dictionary of formatting templates by type
        """
        templates = {
            "report": """
# {title}

## Executive Summary
{executive_summary}

## Introduction
{introduction}

## Key Findings
{key_findings}

## Analysis
{analysis}

## Recommendations
{recommendations}

## Conclusion
{conclusion}
""",
            "guide": """
# {title}

## Overview
{overview}

## Prerequisites
{prerequisites}

## Step-by-Step Instructions
{instructions}

## Tips and Best Practices
{tips}

## Troubleshooting
{troubleshooting}

## Conclusion
{conclusion}
""",
            "analysis": """
# {title}

## Problem Statement
{problem_statement}

## Methodology
{methodology}

## Analysis
{analysis}

## Results
{results}

## Discussion
{discussion}

## Conclusion
{conclusion}
""",
            "comparison": """
# {title}

## Overview
{overview}

## Criteria for Comparison
{criteria}

## Option A: {option_a_name}
{option_a_details}

## Option B: {option_b_name}
{option_b_details}

## Comparison Table
{comparison_table}

## Recommendation
{recommendation}
""",
            "default": """
# {title}

## Introduction
{introduction}

## Main Content
{main_content}

## Conclusion
{conclusion}
"""
        }
        return templates
        
    def compile_manual(self, 
                      topic: str, 
                      contributions: List[str], 
                      additional_text: str = "",
                      output_format: str = "default") -> str:
        """
        Compile a comprehensive manual from agent contributions.
        
        Args:
            topic: Main topic/question
            contributions: List of agent contributions
            additional_text: Additional context or information
            output_format: Desired output format (report, guide, analysis, comparison, default)
            
        Returns:
            Compiled manual text
        """
        with LogContext("compile_manual", topic=topic, format=output_format):
            with PerformanceTimer("compile_manual", threshold_ms=1000):
                # Select appropriate template
                template_key = output_format.lower() if output_format.lower() in self.formatting_templates else "default"
                template = self.formatting_templates[template_key]
                
                # Build compilation prompt
                prompt = self._build_compilation_prompt(topic, contributions, additional_text, template_key)
                
                # Generate compiled manual
                compiled_text = self._generate_content(prompt)
                
                # Store in history
                self.compilation_history.append({
                    "topic": topic,
                    "num_contributions": len(contributions),
                    "output_format": output_format,
                    "timestamp": str(logger.record["time"].datetime.isoformat())
                })
                
                return compiled_text
                
    def _build_compilation_prompt(self, 
                                topic: str, 
                                contributions: List[str], 
                                additional_text: str,
                                format_type: str) -> str:
        """
        Build a prompt for compiling contributions into a manual.
        
        Args:
            topic: Main topic/question
            contributions: List of agent contributions
            additional_text: Additional context or information
            format_type: Format type (report, guide, etc.)
            
        Returns:
            Compilation prompt
        """
        # Format contributions
        formatted_contributions = ""
        for i, contribution in enumerate(contributions):
            formatted_contributions += f"\n\nContribution {i+1}:\n{contribution}"
            
        # Describe desired format based on type
        format_description = ""
        if format_type == "report":
            format_description = (
                "Create a formal report with an executive summary, introduction, key findings, "
                "detailed analysis, actionable recommendations, and conclusion. Use professional "
                "language and structure information logically with clear section headings."
            )
        elif format_type == "guide":
            format_description = (
                "Create a comprehensive how-to guide with overview, prerequisites, clear step-by-step "
                "instructions, helpful tips, troubleshooting advice, and conclusion. Make instructions "
                "actionable and easy to follow with numbered steps and visual descriptions."
            )
        elif format_type == "analysis":
            format_description = (
                "Create an analytical document with problem statement, methodology, detailed analysis, "
                "results, discussion of implications, and conclusion. Emphasize logical reasoning and "
                "evidence-based conclusions, organizing complex information clearly."
            )
        elif format_type == "comparison":
            format_description = (
                "Create a comparative analysis with overview, comparison criteria, detailed assessment "
                "of each option, comparison table, and final recommendation. Present balanced analysis "
                "with clear contrasts between options."
            )
        else:
            format_description = (
                "Create a well-structured document with introduction, main content divided into "
                "logical sections, and conclusion. Use clear headings, organize information "
                "coherently, and ensure content flows logically."
            )
            
        # Build the prompt
        prompt = (
            f"Using the following contributions from experts and the additional text provided, "
            f"compile a comprehensive document on the topic.\n\n"
            f"Topic: {topic}\n\n"
            f"Contributions:{formatted_contributions}\n\n"
            f"Additional Text:\n{additional_text}\n\n"
            f"Output Format Instructions:\n{format_description}\n\n"
            f"Important Guidelines:\n"
            f"1. Synthesize all contributions into a cohesive whole\n"
            f"2. Eliminate redundancies while preserving unique insights\n"
            f"3. Ensure logical organization with clear structure\n"
            f"4. Use markdown formatting for headings and sections\n"
            f"5. Create a comprehensive document that directly addresses the topic\n"
            f"6. Do not mention the individual contributors or that this was compiled from multiple sources\n"
            f"7. Make sure each section is detailed and substantive\n"
            f"8. Use examples, analogies, or illustrations where appropriate\n\n"
            f"Begin your comprehensive document now."
        )
        
        return prompt
        
    def _generate_content(self, prompt: str) -> str:
        """
        Generate content using the LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated content
        """
        try:
            messages = [{'role': 'user', 'content': prompt}]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 4000  # Higher token limit for comprehensive content
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.exception(f"Error generating content: {e}")
            return f"Error generating content: {str(e)}"
            
    def analyze_content_type(self, topic: str, contributions: List[str]) -> str:
        """
        Analyze content to determine optimal formatting template.
        
        Args:
            topic: Main topic/question
            contributions: List of agent contributions
            
        Returns:
            Suggested format type
        """
        # Create analysis prompt
        combined_text = " ".join(contributions)
        sample_text = combined_text[:5000] if len(combined_text) > 5000 else combined_text
        
        prompt = (
            f"Analyze the following content and determine the most appropriate format "
            f"from these options: report, guide, analysis, comparison, or default.\n\n"
            f"Topic: {topic}\n\n"
            f"Sample Content:\n{sample_text}\n\n"
            f"Respond with a JSON object with the format: {{\"format\": \"chosen_format\", \"reason\": \"explanation\"}}"
        )
        
        try:
            messages = [{'role': 'user', 'content': prompt}]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.3}  # Lower temperature for analytical task
            )
            
            result_text = response['message']['content'].strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    return result.get("format", "default")
                except:
                    pass
                    
            # Fallback: simple keyword matching
            result_text = result_text.lower()
            if "how-to" in result_text or "step" in result_text or "instruction" in result_text:
                return "guide"
            elif "compare" in result_text or "option" in result_text or "versus" in result_text:
                return "comparison"
            elif "analyze" in result_text or "research" in result_text or "study" in result_text:
                return "analysis"
            elif "report" in result_text or "findings" in result_text or "executive" in result_text:
                return "report"
                
            return "default"
            
        except Exception as e:
            logger.exception(f"Error analyzing content type: {e}")
            return "default"
            
    def enhance_with_visuals(self, content: str, topic: str) -> str:
        """
        Enhance content with suggested visualizations or diagrams.
        
        Args:
            content: Compiled content
            topic: Main topic
            
        Returns:
            Content with visualization suggestions
        """
        # Create analysis prompt
        prompt = (
            f"Analyze the following content and suggest appropriate visualizations "
            f"or diagrams that would enhance understanding. Add markdown placeholders "
            f"for the suggested visualizations at appropriate locations in the content.\n\n"
            f"Topic: {topic}\n\n"
            f"Content:\n{content}\n\n"
            f"Instructions:\n"
            f"1. Identify 2-3 key points that would benefit from visualization\n"
            f"2. For each, add a visualization suggestion in the format:\n"
            f"   ```diagram\n"
            f"   [Type: e.g., flowchart, comparison table, concept map]\n"
            f"   [Description: Brief description of what the diagram should show]\n"
            f"   [Key Elements: List of important elements to include]\n"
            f"   ```\n"
            f"3. Place each suggestion at an appropriate location in the content\n"
            f"4. Do not otherwise change the content significantly\n\n"
            f"Return the enhanced content with visualization suggestions integrated."
        )
        
        try:
            messages = [{'role': 'user', 'content': prompt}]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.5}
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.exception(f"Error enhancing with visuals: {e}")
            return content  # Return original content if enhancement fails
            
    def simplify_complex_content(self, 
                               content: str, 
                               target_complexity: str = "medium") -> str:
        """
        Simplify complex content to target reading level.
        
        Args:
            content: Content to simplify
            target_complexity: Target complexity (simple, medium, advanced)
            
        Returns:
            Simplified content
        """
        # Define complexity levels
        complexity_guide = {
            "simple": "8-10 grade reading level, short sentences, common vocabulary",
            "medium": "High school to early college reading level, varied sentence structure",
            "advanced": "College/professional reading level, specialized vocabulary acceptable"
        }
        
        complexity_description = complexity_guide.get(
            target_complexity.lower(), 
            complexity_guide["medium"]
        )
        
        # Create simplification prompt
        prompt = (
            f"Revise the following content to achieve a {target_complexity} complexity level "
            f"({complexity_description}), while preserving all key information and maintaining "
            f"the original structure.\n\n"
            f"Content:\n{content}\n\n"
            f"Guidelines:\n"
            f"1. Maintain all section headings and organizational structure\n"
            f"2. Preserve all key information and concepts\n"
            f"3. Adjust vocabulary and sentence complexity to target level\n"
            f"4. Break down complex explanations into more digestible parts if needed\n"
            f"5. Ensure content flows logically and maintains coherence\n\n"
            f"Return the revised content at {target_complexity} complexity level."
        )
        
        try:
            messages = [{'role': 'user', 'content': prompt}]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.4}
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.exception(f"Error simplifying content: {e}")
            return content  # Return original content if simplification fails
            
    def generate_executive_summary(self, content: str, max_words: int = 200) -> str:
        """
        Generate an executive summary of the content.
        
        Args:
            content: Source content
            max_words: Maximum length in words
            
        Returns:
            Executive summary
        """
        # Create summary prompt
        prompt = (
            f"Generate a concise executive summary of the following content "
            f"in no more than {max_words} words. Capture the most important points, "
            f"key findings, and main conclusions.\n\n"
            f"Content:\n{content}\n\n"
            f"Guidelines:\n"
            f"1. Focus on the most critical information\n"
            f"2. Maintain a professional, objective tone\n"
            f"3. Structure with clear logical flow\n"
            f"4. Include key findings and conclusions\n"
            f"5. Stay under {max_words} words\n\n"
            f"Executive Summary:"
        )
        
        try:
            messages = [{'role': 'user', 'content': prompt}]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.4}
            )
            
            summary = response['message']['content'].strip()
            
            # Check word count
            words = summary.split()
            if len(words) > max_words * 1.1:  # Allow 10% margin
                summary = " ".join(words[:max_words]) + "..."
                
            return summary
            
        except Exception as e:
            logger.exception(f"Error generating executive summary: {e}")
            return "Executive summary generation failed."
            
    def extract_key_points(self, content: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from compiled content.
        
        Args:
            content: Source content
            num_points: Number of key points to extract
            
        Returns:
            List of key points
        """
        # Create key points extraction prompt
        prompt = (
            f"Extract exactly {num_points} key points from the following content. "
            f"Identify the most important ideas, findings, or conclusions. "
            f"Phrase each point as a clear, standalone statement.\n\n"
            f"Content:\n{content}\n\n"
            f"Guidelines:\n"
            f"1. Focus on the most significant ideas\n"
            f"2. Ensure points are distinct from each other\n"
            f"3. Write concise, clear statements\n"
            f"4. Maintain original meaning and intent\n"
            f"5. Extract exactly {num_points} points\n\n"
            f"Format your response as a JSON array of strings with exactly {num_points} items."
        )
        
        try:
            messages = [{'role': 'user', 'content': prompt}]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.3}
            )
            
            result_text = response['message']['content'].strip()
            
            # Extract JSON array
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                try:
                    points = json.loads(json_match.group(0))
                    # Ensure we have exactly the right number of points
                    while len(points) > num_points:
                        points.pop()
                    while len(points) < num_points and points:
                        # Split the longest point if we need more
                        longest_idx = max(range(len(points)), key=lambda i: len(points[i]))
                        parts = points[longest_idx].split('. ', 1)
                        if len(parts) > 1:
                            points[longest_idx] = parts[0] + '.'
                            points.append(parts[1].strip())
                        else:
                            # Can't split further, duplicate a point
                            points.append(f"Additional consideration: {points[0]}")
                    return points
                except:
                    pass
                    
            # Fallback: manual extraction
            lines = result_text.split('\n')
            points = []
            
            for line in lines:
                line = line.strip()
                # Look for numbered or bullet points
                if re.match(r'^\d+[\.\)]|^[-*•]\s', line):
                    # Clean up the line
                    clean_line = re.sub(r'^\d+[\.\)]|^[-*•]\s', '', line).strip()
                    points.append(clean_line)
                    if len(points) >= num_points:
                        break
                        
            # If still not enough points, split text into sentences
            if len(points) < num_points:
                sentences = re.split(r'[.!?]+', result_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 20 and sentence not in points:
                        points.append(sentence)
                        if len(points) >= num_points:
                            break
                            
            return points[:num_points]
            
        except Exception as e:
            logger.exception(f"Error extracting key points: {e}")
            return [f"Key point {i+1} extraction failed." for i in range(num_points)]
