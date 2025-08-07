#!/usr/bin/env python3
"""
Local security check script for AI Safety Newsletter Agent.
Runs basic security checks before committing code.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SecurityChecker:
    """Run security checks on the codebase."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues: List[Dict] = []
        
    def run_all_checks(self) -> Tuple[bool, List[Dict]]:
        """Run all security checks.
        
        Returns:
            Tuple of (success, issues_list)
        """
        print("🔒 Running security checks...\n")
        
        # Run individual checks
        checks = [
            ("Hardcoded secrets", self.check_hardcoded_secrets),
            ("Dangerous patterns", self.check_dangerous_patterns),
            ("File permissions", self.check_file_permissions),
            ("Dependencies", self.check_dependencies),
            ("Configuration", self.check_configuration),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"Running {check_name} check...")
            passed = check_func()
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}\n")
            
            if not passed:
                all_passed = False
        
        return all_passed, self.issues
    
    def check_hardcoded_secrets(self) -> bool:
        """Check for hardcoded secrets in Python files."""
        dangerous_patterns = [
            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
            r'password\s*=\s*["\'][^"\']{3,}["\']',
            r'secret\s*=\s*["\'][^"\']{10,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']',
            r'OPENAI_API_KEY\s*=\s*["\'][^"\']{10,}["\']',
            r'google.*api.*key\s*=\s*["\'][^"\']{10,}["\']',
        ]
        
        import re
        
        python_files = list(self.project_root.rglob("*.py"))
        issues_found = False
        
        for file_path in python_files:
            if "test_" in file_path.name or ".venv" in str(file_path) or "security-check.py" in file_path.name:
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                
                for pattern in dangerous_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self.issues.append({
                            'type': 'hardcoded_secret',
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'message': f'Potential hardcoded secret: {match.group()[:20]}...',
                            'severity': 'high'
                        })
                        issues_found = True
                        
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        return not issues_found
    
    def check_dangerous_patterns(self) -> bool:
        """Check for dangerous code patterns."""
        dangerous_patterns = [
            (r'subprocess\.call\([^)]*shell=True', 'Shell injection risk'),
            (r'os\.system\(', 'Command injection risk'),
            (r'eval\(', 'Code injection risk'),
            (r'exec\(', 'Code injection risk'),
            (r'pickle\.loads?\(', 'Deserialization vulnerability'),
            (r'yaml\.load\([^)]*Loader=(?!SafeLoader)', 'Unsafe YAML loading'),
        ]
        
        import re
        
        python_files = list(self.project_root.rglob("*.py"))
        issues_found = False
        
        for file_path in python_files:
            if "test_" in file_path.name or ".venv" in str(file_path) or "security-check.py" in file_path.name:
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8')
                
                for pattern, message in dangerous_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Skip false positives in security.py (patterns are strings)
                        if 'security.py' in str(file_path) and ('eval(' in match.group() or 'exec(' in match.group()):
                            # Check if this is in a string literal (basic check)
                            line_start = content.rfind('\n', 0, match.start()) + 1
                            line_end = content.find('\n', match.end())
                            if line_end == -1:
                                line_end = len(content)
                            line_content = content[line_start:line_end]
                            
                            # Skip if it's in a string literal or comment
                            if ("'" in line_content and line_content.count("'") >= 2) or \
                               ('"' in line_content and line_content.count('"') >= 2) or \
                               line_content.strip().startswith('#'):
                                continue
                        
                        self.issues.append({
                            'type': 'dangerous_pattern',
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'message': message,
                            'code': match.group(),
                            'severity': 'high'
                        })
                        issues_found = True
                        
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        return not issues_found
    
    def check_file_permissions(self) -> bool:
        """Check for overly permissive file permissions."""
        issues_found = False
        
        # Check for world-writable files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = stat.st_mode
                    
                    # Check if world-writable (others have write permission)
                    if mode & 0o002:
                        self.issues.append({
                            'type': 'file_permissions',
                            'file': str(file_path.relative_to(self.project_root)),
                            'message': 'File is world-writable',
                            'severity': 'medium'
                        })
                        issues_found = True
                        
                    # Check if group-writable for sensitive files
                    if file_path.suffix in ['.py', '.yaml', '.yml', '.env'] and mode & 0o020:
                        self.issues.append({
                            'type': 'file_permissions',
                            'file': str(file_path.relative_to(self.project_root)),
                            'message': 'Sensitive file is group-writable',
                            'severity': 'medium'
                        })
                        issues_found = True
                        
                except (OSError, PermissionError):
                    pass  # Skip files we can't access
        
        return not issues_found
    
    def check_dependencies(self) -> bool:
        """Check for known vulnerable dependencies."""
        # This is a basic check - in production, use tools like safety or pip-audit
        pyproject_path = self.project_root / "pyproject.toml"
        
        if not pyproject_path.exists():
            return True
        
        try:
            import tomli
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomli.load(f)
        except ImportError:
            # If tomli not available, skip this check
            return True
        except Exception:
            return True
        
        # Basic check for old/vulnerable packages (this is just an example)
        dependencies = pyproject_data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        
        # Known vulnerable versions (example - update with real data)
        vulnerable_packages = {
            "requests": ["<2.20.0"],
            "urllib3": ["<1.24.2"],
            "pyyaml": ["<5.1"],
        }
        
        issues_found = False
        for pkg_name, version_spec in dependencies.items():
            if pkg_name in vulnerable_packages:
                # This is a simplified check - real implementation would parse version constraints
                self.issues.append({
                    'type': 'vulnerable_dependency',
                    'package': pkg_name,
                    'version': version_spec,
                    'message': f'Package {pkg_name} may have known vulnerabilities',
                    'severity': 'medium'
                })
                issues_found = True
        
        return not issues_found
    
    def check_configuration(self) -> bool:
        """Check for insecure configuration patterns."""
        config_files = [
            self.project_root / "aisafety_news" / "config.py",
            self.project_root / ".env.example",
        ]
        
        issues_found = False
        
        for config_file in config_files:
            if not config_file.exists():
                continue
                
            try:
                content = config_file.read_text(encoding='utf-8')
                
                # Check for insecure defaults
                insecure_patterns = [
                    (r'https_only.*=.*False', 'HTTPS enforcement disabled'),
                    (r'debug.*=.*True', 'Debug mode enabled'),
                    (r'ssl_verify.*=.*False', 'SSL verification disabled'),
                    (r'timeout.*=.*None', 'No timeout configured'),
                ]
                
                import re
                for pattern, message in insecure_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.issues.append({
                            'type': 'insecure_config',
                            'file': str(config_file.relative_to(self.project_root)),
                            'message': message,
                            'severity': 'medium'
                        })
                        issues_found = True
                        
            except Exception as e:
                print(f"Warning: Could not read {config_file}: {e}")
        
        return not issues_found
    
    def generate_report(self) -> str:
        """Generate a security report."""
        if not self.issues:
            return "✅ No security issues found!"
        
        report = f"🔒 Security Issues Found: {len(self.issues)}\n\n"
        
        # Group by severity
        by_severity = {}
        for issue in self.issues:
            severity = issue.get('severity', 'unknown')
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)
        
        for severity in ['high', 'medium', 'low']:
            if severity in by_severity:
                report += f"## {severity.upper()} Severity ({len(by_severity[severity])} issues)\n\n"
                
                for issue in by_severity[severity]:
                    report += f"- **{issue.get('file', 'Unknown file')}**"
                    if 'line' in issue:
                        report += f" (line {issue['line']})"
                    report += f": {issue['message']}\n"
                    
                    if 'code' in issue:
                        report += f"  ```\n  {issue['code']}\n  ```\n"
                
                report += "\n"
        
        return report


def main():
    """Run security checks from command line."""
    project_root = Path(__file__).parent.parent
    checker = SecurityChecker(project_root)
    
    success, issues = checker.run_all_checks()
    
    print("\n" + "="*50)
    print(checker.generate_report())
    
    if not success:
        print("❌ Security checks failed! Please fix the issues above.")
        sys.exit(1)
    else:
        print("✅ All security checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()