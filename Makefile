.PHONY: clean

clean:
	cargo clean
	rm -rf build/ dist/ .pytest_cache/ htmlcov/ *.egg-info
	rm -f .coverage .coverage.*
	find python tests -name __pycache__ -type d -exec rm -rf {} +
	find python -name "*.so" -delete -o -name "*.pyd" -delete
