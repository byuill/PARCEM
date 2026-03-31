## 2025-02-28 - [Notebook optimization learnings]
**Learning:** Hoisting variables outside loops requires extreme care. When modifying Jupyter Notebook structures programmatically (e.g. parsing JSON), remember to ensure required outer-scope variables remain defined.
**Action:** Always test both paths of conditionally executed code (e.g., setting different config variables like `TRANSPORT_MODE = 'Blended'`) when extracting variables from inner loops to outer loops.
