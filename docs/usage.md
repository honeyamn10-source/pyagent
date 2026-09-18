
## Cancellation & escalation

Agents that hit their step budget call `ctx.cancel()`; set `escalation` on the
run to hand off to a human reviewer. Example:

```python
result = agent.run(escalation="reviewer@ops")
```
