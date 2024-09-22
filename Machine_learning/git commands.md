
# Git Commands Cheat Sheet

### 1. **Initializing a Git Repository**

- To create a new Git repository in your project directory:

  ```bash
  git init
  ```

### 2. **Cloning a Repository**

- To clone an existing repository from a remote source (e.g., GitHub):

  ```bash
  git clone [repository URL]
  ```

### 3. **Checking the Status**

- To see the current state of your working directory and staging area:

  ```bash
  git status
  ```

### 4. **Staging Changes**

- To stage individual files:

  ```bash
  git add filename
  ```

- To stage all changes in the current directory:

  ```bash
  git add .
  ```

### 5. **Committing Changes**

- To commit all staged files with a message:

  ```bash
  git commit -m "Commit message"
  ```

### 6. **Viewing Commit History**

- To view the commit history of the repository:

  ```bash
  git log
  ```

  - You can use `git log --oneline` for a more concise view.

### 7. **Branching**

- **Create a new branch**:

  ```bash
  git branch branch_name
  ```
  
- **Switch to a different branch**:

  ```bash
  git checkout branch_name
  ```
  
- **Create and switch to a new branch in one command**:

  ```bash
  git checkout -b branch_name
  ```

### 8. **Merging Branches**

- To merge a branch into the current branch:

  ```bash
  git merge branch_name
  ```

### 9. **Pushing Changes to Remote Repository**

- To push your local commits to a remote repository (e.g., GitHub):

  ```bash
  git push origin branch_name
  ```

### 10. **Pulling Updates from Remote Repository**

- To fetch and merge changes from a remote repository to your local branch:

  ```bash
  git pull origin branch_name
  ```

### 11. **Undoing Changes**

- **Unstage a file**:

  ```bash
  git reset filename
  ```

- **Revert to the last commit, discarding all changes**:

  ```bash
  git reset --hard HEAD
  ```

### 12. **Deleting Branches**

- To delete a local branch:

  ```bash
  git branch -d branch_name
  ```


## Get back repo? --> : [https://github.com/3XCeptional/Ml-and-Data-Science-Notes-Notebooks/](https://github.com/3XCeptional/Ml-and-Data-Science-Notes-Notebooks/)
