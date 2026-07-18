.PHONY: clean clean-results clean-logs clean-cache

clean: clean-results clean-logs clean-cache

clean-results:
	rm -f result/*.png result/*.jpg result/*.html

clean-logs:
	rm -f log/*

clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
