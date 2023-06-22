For clean submit without large files, please enter do the following
1. Install jq
2. Write the following lines into .gitconfig:
[core]
        excludesfile = ~/.gitignore
        attributesfile = ~/.gitattributes_global
[filter "nbstrip_full"]
        clean = "jq --indent 1 \
            '(.cells[] | select(has(\"outputs\")) | .outputs) = []  \
                | (.cells[] | select(has(\"execution_count\")) | .execution_count) = null  \
                | .metadata = {\"language_info\": {\"name\": \"python\", \"pygments_lexer\": \"ipython3\"}} \
                | .cells[].metadata = {} \
                '"
        smudge = cat
        required = true
3. Write the following lines into .gitattributes_global
*.ipynb filter=nbstrip_full
4. Write the following lines into .gitignore
# Compiled source #
###################
*.com
*.class
*.dll
*.exe
*.o
*.so
*.pth

# Caches #
##########
__pycache__
.*cache
.Trash*

# Packages #
############
# it's better to unpack these files and commit the raw source
# git has its own built in compression methods
*.7z
*.dmg
*.gz
*.iso
*.jar
*.rar
*.tar
*.zip

# Tools #
#########
.hypothesis
.ipynb_checkpoints
