User: Hi!

Assistant: Hello, how can I assist you?

User: Below I will show a dialogue between {0} and {1} on trade of stars, coins and cards (of 3 types: rock, paper, scissors). Please extract the items **{0}** may want to take out to trade in JSON format.
Here is an example JSON output:
```json
{{
    "star": "number of stars",
    "coin": "number of coins",
    "rock": "number of rock cards",
    "paper": "number of paper cards",
    "scissors": "number of scissor cards"
}}
```
And here is the dialogue:
```
{2}
```
Remember to extract what **{0}** wants to take out to trade. If the subject doesn't have strong trade intention, keep those fields zero.

Assistant: Sure, based on the dialogue provided, the items {0} wants to take out are
```json
