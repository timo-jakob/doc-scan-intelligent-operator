# Renovate Setup Guide

This repository uses [Renovate](https://docs.renovatebot.com/) for automated dependency management.

## Overview

Renovate automatically:
- 🔍 Detects outdated dependencies in `requirements.txt` and `requirements-dev.txt`
- 📝 Creates Pull Requests with dependency updates
- 🏷️ Labels PRs appropriately (production, development, MLX, security, etc.)
- 📊 Maintains a Dependency Dashboard issue
- ⚡ Groups related updates together

## Configuration Files

### 1. `renovate.json` (Root)
Main configuration file with core settings:
- Scheduled updates: Mondays before 10am (Europe/Berlin)
- Grouped updates by type (production vs development)
- Special handling for MLX packages (critical for Apple Silicon)
- Security vulnerability alerts
- Major updates wait 7 days for stability

### 2. `.github/renovate.json5` (GitHub-specific)
Additional GitHub-specific settings:
- Dependency Dashboard enabled
- Semantic commit messages
- Pull Request formatting
- Branch and PR limits

### 3. `.github/CODEOWNERS`
Ensures you're automatically requested as reviewer on dependency PRs

### 4. `.github/workflows/dependency-test.yml`
GitHub Actions workflow that:
- Tests dependency updates on macOS (Apple Silicon compatibility)
- Verifies MLX installation
- Runs security audits
- Checks Python version compatibility

## How to Use

### Initial Setup

1. **Enable Renovate App** (if not already done):
   - Visit https://github.com/apps/renovate
   - Install for your repository
   - Renovate will automatically detect `renovate.json`

2. **First Run**:
   - Renovate creates an "Configure Renovate" onboarding PR
   - Review and merge to activate
   - Dependency Dashboard issue will be created

### Managing Updates

#### Dependency Dashboard
Renovate creates a pinned issue showing:
- All available updates
- Rate-limited updates
- Pending updates
- Any configuration warnings

Access it: Look for issue titled "🤖 Dependency Updates Dashboard"

#### Pull Request Workflow

1. **Renovate creates PR**:
   ```
   chore(deps): update mlx to v0.22.0
   ```

2. **Automatic checks run**:
   - Installation test on macOS
   - MLX compatibility verification
   - Security audit
   - Linting checks

3. **Your review**:
   - Check the PR description for changelogs
   - Review breaking changes
   - Look at test results
   - Manually test if needed (especially MLX packages)

4. **Approve and merge**:
   - Once satisfied, approve the PR
   - Merge when ready
   - Renovate will rebase other PRs if needed

### Update Grouping

Dependencies are grouped as follows:

#### Production Dependencies
- All packages in `requirements.txt`
- Label: `dependencies`, `production`
- PR title: "chore(deps): update production dependencies"

#### Development Dependencies
- All packages in `requirements-dev.txt`
- Label: `dependencies`, `development`
- PR title: "chore(deps): update development dependencies"

#### MLX Packages (Special handling)
- `mlx`, `mlx-lm`, `mlx-vlm`
- Labels: `dependencies`, `mlx`, `critical`
- Separate PR for visibility
- **Critical**: These are essential for Apple Silicon functionality

#### Security Updates
- Labeled: `security`, `vulnerability`
- Higher priority (processed first)
- May appear as separate PRs

### Update Types

#### Patch Updates (0.0.X)
- Bug fixes and security patches
- Generally safe to merge
- Example: `1.2.3` → `1.2.4`

#### Minor Updates (0.X.0)
- New features, backwards compatible
- Review changelog recommended
- Example: `1.2.3` → `1.3.0`

#### Major Updates (X.0.0)
- Breaking changes possible
- Waits 7 days before PR creation
- **Requires careful review and testing**
- Example: `1.2.3` → `2.0.0`

## Configuration Customization

### Change Update Schedule

Edit `renovate.json`:
```json
{
  "schedule": [
    "before 10am on monday"  // Change to your preference
  ]
}
```

Options:
- `"before 10am on monday"`
- `"after 10pm every weekday"`
- `"every weekend"`
- `["on monday", "on wednesday"]`

### Change PR Limits

Edit `renovate.json`:
```json
{
  "prConcurrentLimit": 5,  // Max open PRs at once
  "branchConcurrentLimit": 10  // Max branches
}
```

### Ignore Specific Packages

Edit `.github/renovate.json5`:
```json
{
  "ignoreDeps": [
    "package-name-to-ignore"
  ]
}
```

### Auto-merge (Not Recommended for Production)

If you want to auto-merge specific updates:
```json
{
  "packageRules": [
    {
      "matchUpdateTypes": ["patch"],
      "matchPackagePatterns": ["^pytest"],
      "automerge": true
    }
  ]
}
```

## Monitoring

### Check Renovate Status

1. **Dependency Dashboard**: Check the pinned issue
2. **Pull Requests**: Filter by label `dependencies`
3. **Renovate Logs**: Available in PR comments

### Common Issues

#### No PRs created
- Check Dependency Dashboard for rate limits
- Verify Renovate app is installed
- Check `renovate.json` for syntax errors

#### PR fails checks
- Review GitHub Actions workflow logs
- Common causes:
  - Installation fails on macOS
  - MLX incompatibility
  - Breaking changes in major updates

#### Too many PRs
- Reduce `prConcurrentLimit`
- Adjust grouping rules
- Use `minimumReleaseAge` to delay updates

## Best Practices

### For MLX Packages
1. **Never auto-merge** - These are critical
2. **Test locally** before merging:
   ```bash
   pip install -r requirements.txt
   python -c "import mlx.core as mx; print(mx.__version__)"
   docscan test_invoice.pdf --dry-run
   ```
3. **Check compatibility** with Apple Silicon

### For Security Updates
1. **Review CVE details** in PR description
2. **Merge promptly** after verification
3. **Monitor for regression**

### For Major Updates
1. **Read release notes carefully**
2. **Test thoroughly**:
   - Run full test suite
   - Test invoice processing
   - Check for breaking changes
3. **Update code if needed** before merging
4. **Consider in separate branch** for extensive testing

## Maintenance

### Monthly Review
- Review Dependency Dashboard
- Check for ignored/postponed updates
- Update Renovate configuration if needed

### After Updates
- Monitor application behavior
- Check logs for warnings
- Run invoice processing on test documents

## Support

- **Renovate Docs**: https://docs.renovatebot.com/
- **GitHub Issues**: Check repository issues for problems
- **Configuration Validator**: Use online validator for `renovate.json`

## Emergency: Pause Renovate

If needed, temporarily pause Renovate:

1. Close all open Renovate PRs
2. Add to `renovate.json`:
   ```json
   {
     "enabled": false
   }
   ```
3. Commit and push
4. Re-enable when ready by removing or setting to `true`
