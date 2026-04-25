# deploy_to_hf.ps1
# This script deploys the current state of the repository to Hugging Face Spaces
# It bypasses the "binary files in history" error by creating a clean orphan branch.

Write-Host "Starting deployment to Hugging Face Space..." -ForegroundColor Cyan

# 1. Create a new orphan branch (history-free)
git checkout --orphan hf-deploy-clean

# 2. Remove all cached files to start completely fresh
git rm -rf --cached . | Out-Null

# 3. Add .gitattributes first so LFS rules are applied
git add .gitattributes

# 4. Add the rest of the files
git add -A

# 5. Commit the clean state
git commit -m "Clean deploy to HF Space" | Out-Null

# 6. Force push the clean branch to the Hugging Face space's main branch
Write-Host "Pushing to Hugging Face..." -ForegroundColor Yellow
git push space hf-deploy-clean:main -f

# 7. Clean up: Switch back to main and delete the temporary branch
git checkout main 2>&1 | Out-Null
git branch -D hf-deploy-clean 2>&1 | Out-Null

Write-Host "Deployment completed successfully!" -ForegroundColor Green
