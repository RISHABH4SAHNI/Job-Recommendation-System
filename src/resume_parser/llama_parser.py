"""Resume parsing using LlamaIndex"""
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LlamaResumeParser:
    """Parse resumes using LlamaIndex and convert to structured format"""

    def __init__(self):
        try:
            from llama_index import SimpleDirectoryReader, Document
            self.SimpleDirectoryReader = SimpleDirectoryReader
            self.Document = Document
            self.llama_available = True
        except ImportError:
            logger.warning("LlamaIndex not available. Resume parsing will be limited.")
            self.llama_available = False

    def parse_resume_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a single resume file"""
        if not self.llama_available:
            raise ImportError("LlamaIndex is required for resume parsing")

        try:
            # Load document
            documents = self.SimpleDirectoryReader(input_files=[str(file_path)]).load_data()

            if not documents:
                logger.error(f"No content extracted from {file_path}")
                return {}

            # Extract text content
            resume_text = documents[0].text

            # Parse structured information
            parsed_data = self._extract_structured_data(resume_text)

            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing resume {file_path}: {e}")
            return {}

    def parse_resume_directory(self, directory_path: Path) -> pd.DataFrame:
        """Parse all resumes in a directory"""
        if not self.llama_available:
            raise ImportError("LlamaIndex is required for resume parsing")

        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        parsed_resumes = []

        # Supported file extensions
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt'}

        for file_path in directory_path.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                logger.info(f"Parsing resume: {file_path.name}")
                parsed_data = self.parse_resume_from_file(file_path)

                if parsed_data:
                    parsed_data['source_file'] = file_path.name
                    parsed_resumes.append(parsed_data)

        logger.info(f"Parsed {len(parsed_resumes)} resumes from {directory_path}")
        return pd.DataFrame(parsed_resumes)

    def _extract_structured_data(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured data from resume text"""
        # This is a simplified extraction - in practice, you'd use more sophisticated NLP
        data = {
            'Name': self._extract_name(resume_text),
            'Email': self._extract_email(resume_text),
            'Phone': self._extract_phone(resume_text),
            'LinkedIn': self._extract_linkedin(resume_text),
            'Github': self._extract_github(resume_text),
            'Degree': self._extract_degree(resume_text),
            'Major': self._extract_major(resume_text),
            'Year': self._extract_year(resume_text),
            'CGPA': self._extract_cgpa(resume_text),
            'Experiences': self._extract_experiences(resume_text),
            'Projects': self._extract_projects(resume_text),
            'HardSkills': self._extract_skills(resume_text),
            'full_text': resume_text
        }

        return data

    def _extract_name(self, text: str) -> str:
        """Extract name from resume text - simplified implementation"""
        lines = text.split('\n')
        # Assume first non-empty line is the name
        for line in lines:
            if line.strip():
                return line.strip()
        return ""

    def _extract_email(self, text: str) -> str:
        """Extract email from resume text"""
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""

    def _extract_phone(self, text: str) -> str:
        """Extract phone number from resume text"""
        import re
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        matches = re.findall(phone_pattern, text)
        return ''.join(matches[0]) if matches else ""

    def _extract_linkedin(self, text: str) -> str:
        """Extract LinkedIn profile from resume text"""
        import re
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
        return matches[0] if matches else ""

    def _extract_github(self, text: str) -> str:
        """Extract GitHub profile from resume text"""
        import re
        github_pattern = r'github\.com/[\w-]+'
        matches = re.findall(github_pattern, text, re.IGNORECASE)
        return matches[0] if matches else ""

    def _extract_degree(self, text: str) -> str:
        """Extract degree information"""
        degree_keywords = ['B.E.', 'B.Tech', 'M.Tech', 'M.E.', 'Bachelor', 'Master', 'PhD']
        for keyword in degree_keywords:
            if keyword.lower() in text.lower():
                return keyword
        return ""

    def _extract_major(self, text: str) -> str:
        """Extract major/field of study"""
        majors = ['Computer Science', 'Electrical Engineering', 'Mechanical Engineering', 
                 'Chemical Engineering', 'Data Science', 'Information Technology']
        for major in majors:
            if major.lower() in text.lower():
                return major
        return ""

    def _extract_year(self, text: str) -> str:
        """Extract graduation year"""
        import re
        year_pattern = r'20\d{2}'
        matches = re.findall(year_pattern, text)
        return max(matches) if matches else ""

    def _extract_cgpa(self, text: str) -> str:
        """Extract CGPA"""
        import re
        cgpa_pattern = r'(\d+\.\d{2})\s*(?:cgpa|gpa|cpi)'
        matches = re.findall(cgpa_pattern, text, re.IGNORECASE)
        return matches[0] if matches else ""

    def _extract_experiences(self, text: str) -> str:
        """Extract work experiences - simplified"""
        # In practice, this would use more sophisticated NLP
        experience_keywords = ['experience', 'intern', 'work', 'job', 'position']
        relevant_lines = []

        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in experience_keywords):
                relevant_lines.append(line.strip())

        return '\n'.join(relevant_lines)

    def _extract_projects(self, text: str) -> str:
        """Extract projects - simplified"""
        project_keywords = ['project', 'built', 'developed', 'created']
        relevant_lines = []

        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in project_keywords):
                relevant_lines.append(line.strip())

        return '\n'.join(relevant_lines)

    def _extract_skills(self, text: str) -> str:
        """Extract technical skills"""
        # Common technical skills
        skills = ['python', 'java', 'javascript', 'react', 'node.js', 'sql', 'mongodb',
                 'machine learning', 'deep learning', 'ai', 'data science', 'aws', 'docker']

        found_skills = []
        text_lower = text.lower()

        for skill in skills:
            if skill in text_lower:
                found_skills.append({'skill': skill, 'percentage': 80})  # Default confidence

        return json.dumps(found_skills)

    def convert_json_to_csv(self, json_file_path: Path, output_csv_path: Path):
        """Convert JSON format resume data to CSV"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            # Convert to DataFrame and save
            df = pd.DataFrame(data)
            df.to_csv(output_csv_path, index=False)

            logger.info(f"Converted {json_file_path} to {output_csv_path}")

        except Exception as e:
            logger.error(f"Error converting JSON to CSV: {e}")