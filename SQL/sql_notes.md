
# SELECT Statement Usage

**SELECT** is classified as a Database Query command used to retrieve information from a database table.

There are various forms in which a SELECT statement can be used. Below are examples of its general syntax and practical applications.

## General Syntax of a SELECT Statement
To retrieve data under specified columns from a table (in this case, `TABLE_1`), the syntax is as follows:

```sql
SELECT COLUMN1, COLUMN2, ... FROM TABLE_1;
```

### Retrieving All Columns
To retrieve all columns from a table, use `*` instead of specifying individual column names:

```sql
SELECT * FROM TABLE_1;
```

### Filtering Data with the WHERE Clause
To filter the data based on a condition or predicate, the following syntax can be used:

```sql
SELECT <COLUMNS> FROM TABLE_1 WHERE <predicate>;
```

## SELECT Examples

### Example Database Table: `COUNTRY`
Consider the following table named `COUNTRY`, which contains three columns: `ID`, `Name`, and `CCode` (the two-letter country code).

| ID  | Name                       | CCode |
| --- | -------------------------- | ----- |
| 1   | United States of America    | US    |
| 2   | China                      | CH    |
| 3   | Japan                      | JA    |
| 4   | Germany                    | GE    |
| 5   | India                      | IN    |
| 6   | United Kingdom             | UK    |
| 7   | France                     | FR    |
| 8   | Italy                      | IT    |
| 9   | Canada                     | CA    |
| 10  | Brazil                     | BR    |

### Example #1: Retrieve All Data
The following query retrieves all rows and columns from the `COUNTRY` table:

```sql
SELECT * FROM COUNTRY;
```

Response:

| ID  | Name                       | CCode |
| --- | -------------------------- | ----- |
| 1   | United States of America    | US    |
| 2   | China                      | CH    |
| 3   | Japan                      | JA    |
| 4   | Germany                    | GE    |
| 5   | India                      | IN    |
| 6   | United Kingdom             | UK    |
| 7   | France                     | FR    |
| 8   | Italy                      | IT    |
| 9   | Canada                     | CA    |
| 10  | Brazil                     | BR    |

### Example #2: Retrieve Specific Columns
To retrieve only specific columns, such as `ID` and `Name`, the following query is used:

```sql
SELECT ID, Name FROM COUNTRY;
```

Response:

| ID  | Name                       |
| --- | -------------------------- |
| 1   | United States of America    |
| 2   | China                      |
| 3   | Japan                      |
| 4   | Germany                    |
| 5   | India                      |
| 6   | United Kingdom             |
| 7   | France                     |
| 8   | Italy                      |
| 9   | Canada                     |
| 10  | Brazil                     |

### Example #3: Filtering Rows Based on a Condition
The following query retrieves all columns where the `ID` is less than or equal to 5:

```sql
SELECT * FROM COUNTRY WHERE ID <= 5;
```

Response:

| ID  | Name                       | CCode |
| --- | -------------------------- | ----- |
| 1   | United States of America    | US    |
| 2   | China                      | CH    |
| 3   | Japan                      | JA    |
| 4   | Germany                    | GE    |
| 5   | India                      | IN    |

### Example #4: Filtering Rows by Country Code
To retrieve all columns where the `CCode` is equal to 'CA', use the following query:

```sql
SELECT * FROM COUNTRY WHERE CCode = 'CA';
```

Response:

| ID  | Name   | CCode |
| --- | ------ | ----- |
| 9   | Canada | CA    |

## Summary
- **SELECT** is a Database Query command used to retrieve information from a database table.
- The **SELECT** statement has various forms depending on the specific action required.
- The general syntax retrieves data under the listed columns from a named table.
- Use `*` to retrieve all columns from a table.
- Use the **WHERE** clause to filter data based on a condition.

