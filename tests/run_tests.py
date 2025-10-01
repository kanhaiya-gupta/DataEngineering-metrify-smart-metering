#!/usr/bin/env python3
"""
Test Runner for Metrify Smart Metering Data Pipeline
Comprehensive test execution with different configurations
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for the Metrify Smart Metering Data Pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.reports_dir = self.tests_dir / "reports"
        self.ensure_reports_dir()
    
    def ensure_reports_dir(self):
        """Ensure reports directory exists"""
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_command(self, command: List[str], description: str = "") -> bool:
        """Run a command and return success status"""
        print(f"\n{'='*60}")
        print(f"Running: {description or ' '.join(command)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                check=True,
                capture_output=False,
                text=True
            )
            end_time = time.time()
            print(f"\n✅ {description or 'Command'} completed successfully in {end_time - start_time:.2f}s")
            return True
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            print(f"\n❌ {description or 'Command'} failed after {end_time - start_time:.2f}s")
            print(f"Exit code: {e.returncode}")
            return False
        except Exception as e:
            print(f"\n❌ Error running command: {e}")
            return False
    
    def run_unit_tests(self, verbose: bool = False) -> bool:
        """Run unit tests"""
        command = ["python", "-m", "pytest", "tests/unit/", "-m", "unit"]
        if verbose:
            command.append("-v")
        return self.run_command(command, "Unit Tests")
    
    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests"""
        command = ["python", "-m", "pytest", "tests/integration/", "-m", "integration"]
        if verbose:
            command.append("-v")
        return self.run_command(command, "Integration Tests")
    
    def run_e2e_tests(self, verbose: bool = False) -> bool:
        """Run end-to-end tests"""
        command = ["python", "-m", "pytest", "tests/e2e/", "-m", "e2e"]
        if verbose:
            command.append("-v")
        return self.run_command(command, "End-to-End Tests")
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance tests"""
        command = ["python", "-m", "pytest", "tests/performance/", "-m", "performance"]
        if verbose:
            command.append("-v")
        return self.run_command(command, "Performance Tests")
    
    def run_smoke_tests(self, verbose: bool = False) -> bool:
        """Run smoke tests"""
        command = ["python", "-m", "pytest", "tests/", "-m", "smoke"]
        if verbose:
            command.append("-v")
        return self.run_command(command, "Smoke Tests")
    
    def run_all_tests(self, verbose: bool = False, exclude_slow: bool = False) -> bool:
        """Run all tests"""
        command = ["python", "-m", "pytest", "tests/"]
        if exclude_slow:
            command.extend(["-m", "not slow"])
        if verbose:
            command.append("-v")
        return self.run_command(command, "All Tests")
    
    def run_tests_with_coverage(self, verbose: bool = False) -> bool:
        """Run tests with coverage report"""
        command = [
            "python", "-m", "pytest", "tests/",
            "--cov=src",
            "--cov=presentation",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ]
        if verbose:
            command.append("-v")
        return self.run_command(command, "Tests with Coverage")
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> bool:
        """Run a specific test file or test function"""
        command = ["python", "-m", "pytest", test_path]
        if verbose:
            command.append("-v")
        return self.run_command(command, f"Specific Test: {test_path}")
    
    def run_parallel_tests(self, num_workers: int = 4, verbose: bool = False) -> bool:
        """Run tests in parallel"""
        command = [
            "python", "-m", "pytest", "tests/",
            "-n", str(num_workers)
        ]
        if verbose:
            command.append("-v")
        return self.run_command(command, f"Parallel Tests ({num_workers} workers)")
    
    def run_tests_by_marker(self, marker: str, verbose: bool = False) -> bool:
        """Run tests by marker"""
        command = ["python", "-m", "pytest", "tests/", "-m", marker]
        if verbose:
            command.append("-v")
        return self.run_command(command, f"Tests with marker: {marker}")
    
    def lint_code(self) -> bool:
        """Run code linting"""
        command = ["python", "-m", "flake8", "src/", "tests/"]
        return self.run_command(command, "Code Linting")
    
    def format_code(self) -> bool:
        """Format code"""
        command = ["python", "-m", "black", "src/", "tests/"]
        return self.run_command(command, "Code Formatting")
    
    def type_check(self) -> bool:
        """Run type checking"""
        command = ["python", "-m", "mypy", "src/"]
        return self.run_command(command, "Type Checking")
    
    def security_check(self) -> bool:
        """Run security checks"""
        command = ["python", "-m", "bandit", "-r", "src/"]
        return self.run_command(command, "Security Check")
    
    def generate_test_report(self) -> bool:
        """Generate comprehensive test report"""
        command = [
            "python", "-m", "pytest", "tests/",
            "--html=reports/test_report.html",
            "--self-contained-html",
            "--junitxml=reports/junit.xml"
        ]
        return self.run_command(command, "Test Report Generation")
    
    def clean_reports(self):
        """Clean up old test reports"""
        if self.reports_dir.exists():
            for file in self.reports_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        print("✅ Cleaned up old test reports")
    
    def show_help(self):
        """Show help information"""
        print("""
Metrify Smart Metering Data Pipeline - Test Runner

Usage: python tests/run_tests.py [OPTIONS] [TEST_TYPE]

Test Types:
  unit          Run unit tests only
  integration   Run integration tests only
  e2e           Run end-to-end tests only
  performance   Run performance tests only
  smoke         Run smoke tests only
  all           Run all tests (default)
  coverage      Run tests with coverage report
  parallel      Run tests in parallel
  specific      Run specific test file/function

Options:
  -v, --verbose     Verbose output
  --no-slow         Exclude slow tests
  --workers N       Number of parallel workers (default: 4)
  --marker MARKER   Run tests with specific marker
  --lint           Run code linting
  --format         Format code
  --type-check     Run type checking
  --security       Run security checks
  --clean          Clean old reports
  --help           Show this help

Examples:
  python tests/run_tests.py unit
  python tests/run_tests.py all --verbose
  python tests/run_tests.py performance --no-slow
  python tests/run_tests.py parallel --workers 8
  python tests/run_tests.py --marker database
  python tests/run_tests.py specific tests/unit/core/test_smart_meter.py
  python tests/run_tests.py coverage --verbose
        """)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test Runner for Metrify Smart Metering Data Pipeline")
    parser.add_argument("test_type", nargs="?", default="all", 
                       choices=["unit", "integration", "e2e", "performance", "smoke", "all", "coverage", "parallel", "specific"],
                       help="Type of tests to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-slow", action="store_true", help="Exclude slow tests")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--marker", help="Run tests with specific marker")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--clean", action="store_true", help="Clean old reports")
    parser.add_argument("--test-path", help="Specific test file/function to run")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Clean reports if requested
    if args.clean:
        runner.clean_reports()
        return
    
    # Run code quality checks
    if args.lint:
        success = runner.lint_code()
        if not success:
            sys.exit(1)
    
    if args.format:
        success = runner.format_code()
        if not success:
            sys.exit(1)
    
    if args.type_check:
        success = runner.type_check()
        if not success:
            sys.exit(1)
    
    if args.security:
        success = runner.security_check()
        if not success:
            sys.exit(1)
    
    # Run tests
    success = True
    
    if args.marker:
        success = runner.run_tests_by_marker(args.marker, args.verbose)
    elif args.test_type == "specific":
        if not args.test_path:
            print("❌ --test-path is required for specific test type")
            sys.exit(1)
        success = runner.run_specific_test(args.test_path, args.verbose)
    elif args.test_type == "unit":
        success = runner.run_unit_tests(args.verbose)
    elif args.test_type == "integration":
        success = runner.run_integration_tests(args.verbose)
    elif args.test_type == "e2e":
        success = runner.run_e2e_tests(args.verbose)
    elif args.test_type == "performance":
        success = runner.run_performance_tests(args.verbose)
    elif args.test_type == "smoke":
        success = runner.run_smoke_tests(args.verbose)
    elif args.test_type == "coverage":
        success = runner.run_tests_with_coverage(args.verbose)
    elif args.test_type == "parallel":
        success = runner.run_parallel_tests(args.workers, args.verbose)
    elif args.test_type == "all":
        success = runner.run_all_tests(args.verbose, args.no_slow)
    
    # Generate test report
    if success:
        runner.generate_test_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
