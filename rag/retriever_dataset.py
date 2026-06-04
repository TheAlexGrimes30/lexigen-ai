dataset = [
    {
        "id": 1,
        "question": "что такое обязательство в гражданском праве",
        "relevant_articles": ["307"],
        "hard_negatives": ["322", "413", "438"],
        "category": "definitions"
    },
    {
        "id": 2,
        "question": "кто такой должник и кредитор и какие у них обязанности",
        "relevant_articles": ["307", "308"],
        "hard_negatives": ["399", "322", "413"],
        "category": "definitions"
    },
    {
        "id": 3,
        "question": "какие основания возникновения обязательств по гк рф",
        "relevant_articles": ["307"],
        "hard_negatives": ["438", "452", "449"],
        "category": "general_definition"
    },
    {
        "id": 4,
        "question": "что такое субсидиарная ответственность и когда она применяется",
        "relevant_articles": ["399"],
        "hard_negatives": ["322", "413", "307"],
        "category": "liability"
    },
    {
        "id": 5,
        "question": "когда кредитор может обратиться к субсидиарному должнику",
        "relevant_articles": ["399"],
        "hard_negatives": ["307", "413", "452"],
        "category": "enforcement"
    },
    {
        "id": 6,
        "question": "что происходит если должник и кредитор совпадают в одном лице",
        "relevant_articles": ["413"],
        "hard_negatives": ["399", "322", "438"],
        "category": "termination"
    },
    {
        "id": 7,
        "question": "в каких случаях обязательство прекращается по гк рф",
        "relevant_articles": ["407", "413", "415", "416", "417"],
        "hard_negatives": ["307", "322", "438", "449"],
        "category": "termination"
    },
    {
        "id": 8,
        "question": "что такое акцепт в гражданском праве",
        "relevant_articles": ["438"],
        "hard_negatives": ["307", "452", "449"],
        "category": "contracts"
    },
    {
        "id": 9,
        "question": "может ли молчание считаться акцептом",
        "relevant_articles": ["438"],
        "hard_negatives": ["413", "399", "452"],
        "category": "contracts"
    },
    {
        "id": 10,
        "question": "что считается акцептом в договоре",
        "relevant_articles": ["438"],
        "hard_negatives": ["307", "322", "413"],
        "category": "contracts"
    },
    {
        "id": 11,
        "question": "когда торги могут быть признаны недействительными",
        "relevant_articles": ["449"],
        "hard_negatives": ["438", "452", "307"],
        "category": "auctions"
    },
    {
        "id": 12,
        "question": "какие последствия недействительности торгов",
        "relevant_articles": ["449", "167"],
        "hard_negatives": ["413", "399", "322"],
        "category": "auctions"
    },
    {
        "id": 13,
        "question": "как происходит изменение и расторжение договора",
        "relevant_articles": ["450", "451", "452", "453"],
        "hard_negatives": ["438", "307", "449"],
        "category": "contracts_procedure"
    },
    {
        "id": 14,
        "question": "нужно ли соблюдать досудебный порядок при расторжении договора",
        "relevant_articles": ["452"],
        "hard_negatives": ["413", "399", "322"],
        "category": "procedural"
    },
    {
        "id": 15,
        "question": "что такое солидарная ответственность должников",
        "relevant_articles": ["322", "323", "325"],
        "hard_negatives": ["399", "413", "307"],
        "category": "liability"
    },
    {
        "id": 16,
        "question": "может ли кредитор требовать долг с любого из солидарных должников",
        "relevant_articles": ["322", "323"],
        "hard_negatives": ["399", "452", "438"],
        "category": "liability"
    },
    {
        "id": 17,
        "question": "чем отличается солидарная и субсидиарная ответственность",
        "relevant_articles": ["322", "323", "399"],
        "hard_negatives": ["413", "307", "438"],
        "category": "comparison"
    },
    {
        "id": 18,
        "question": "какие условия акцепта оферты в гражданском праве",
        "relevant_articles": ["438", "435", "440"],
        "hard_negatives": ["452", "322", "399"],
        "category": "contracts"
    },
    {
        "id": 19,
        "question": "что происходит если должник и кредитор совпали",
        "relevant_articles": ["413"],
        "hard_negatives": ["307", "322", "438"],
        "category": "termination"
    },
    {
        "id": 20,
        "question": "как регулируется заключение договора через акцепт",
        "relevant_articles": ["432", "433", "438", "440"],
        "hard_negatives": ["452", "449", "399"],
        "category": "contracts"
    }
]


mini_dataset = dataset[:7]