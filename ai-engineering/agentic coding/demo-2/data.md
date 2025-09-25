# Data Model - 1828 Ledger TEI Demo

## Project Overview
**Goal**: Parse historical ledger → Generate TEI-XML with minimal bookkeeping annotations → Build simple data entry tool

**Scope**: Proof of concept, not production system

## Data Pipeline
```
.txt (raw ledger) → Parser → TEI-XML → React Form → New entries
```

## 1. Input Structure

### Sample Entry
```
Monday September-15th-1828
Derius Drake DR
To ax dlvd By Puffer 9/1    1 50
```

### Fields to Extract
| Field | Example | Pattern |
|-------|---------|---------|
| Date | "Monday September-15th-1828" | Month Day Year |
| Person | "Derius Drake" | Capitalized names |
| Type | "DR" | DR/D/CR/C/Settled |
| Description | "To ax dlvd By Puffer 9/1" | Free text |
| Amount | "1 50" | Dollars Cents |

## 2. TEI-XML Output

### Minimal Structure
```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Ledger 1828</title></titleStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div type="entry">
        <date when="1828-09-15">Monday September-15th-1828</date>
        <persName ana="#bk:Party">Derius Drake</persName>
        <note type="account-type">DR</note>
        <desc>To ax dlvd By Puffer 9/1</desc>
        <measure ana="#bk:Money" quantity="1.50" unit="dollar">1 50</measure>
      </div>
    </body>
  </text>
</TEI>
```

### Bookkeeping Annotations
Only three from DEPCHA ontology:
- `#bk:Party` - Person/entity in transaction
- `#bk:Money` - Monetary value
- `#bk:Commodity` - Physical goods (optional)

## 3. Parser Logic

### JavaScript Implementation
```javascript
function parseEntry(text) {
  const lines = text.split('\n');
  const entry = {};
  
  // Date line
  const dateMatch = lines[0].match(/(\w+)\s+(\w+)-(\d+)\w*-(\d{4})/);
  if (dateMatch) {
    entry.date = formatISO(dateMatch);
    entry.dateText = lines[0];
  }
  
  // Person + Type line  
  const personMatch = lines[1].match(/([A-Z][\w\s]+?)\s+(DR|D|CR|C|Settled)/);
  if (personMatch) {
    entry.person = personMatch[1].trim();
    entry.type = personMatch[2];
  }
  
  // Description + Amount
  const descMatch = lines[2].match(/(.*?)\s+(\d+\s+\d{2}|\d+\.\d{2})/);
  if (descMatch) {
    entry.description = descMatch[1].trim();
    entry.amount = parseAmount(descMatch[2]);
  }
  
  return entry;
}

function toTEI(entry) {
  return `<div type="entry">
    <date when="${entry.date}">${entry.dateText}</date>
    <persName ana="#bk:Party">${entry.person}</persName>
    <note type="account-type">${entry.type}</note>
    <desc>${entry.description}</desc>
    <measure ana="#bk:Money" quantity="${entry.amount}" unit="dollar">
      ${entry.amountText}
    </measure>
  </div>`;
}
```

## 4. React Data Entry Form

### Component Structure
```jsx
function LedgerEntry() {
  const [formData, setFormData] = useState({
    date: '',
    person: '',
    type: 'DR',
    description: '',
    amount: ''
  });
  
  const exportTEI = () => {
    const tei = toTEI(formData);
    downloadXML(tei);
  };
  
  return (
    <div>
      <input type="date" name="date" />
      <input type="text" name="person" placeholder="Person name" />
      <select name="type">
        <option>DR</option>
        <option>CR</option>
        <option>Settled</option>
      </select>
      <input type="text" name="description" placeholder="Description" />
      <input type="number" name="amount" step="0.01" />
      <button onClick={exportTEI}>Export TEI</button>
    </div>
  );
}
```

### Features
- Manual data entry for new records
- Parse pasted text
- Export individual entries as TEI
- No database (browser only)
- No validation beyond basic format

## 5. Currency Handling

### Simple Rules
- "1 50" = $1.50
- "98" = $0.98  
- "6/" = $1.00 (shilling)
- "3/" = $0.50 (shilling)

### Conversion Function
```javascript
function parseAmount(text) {
  if (text.includes('/')) {
    const shillings = parseInt(text);
    return (shillings / 6).toFixed(2);
  }
  if (text.includes(' ')) {
    return text.replace(' ', '.');
  }
  return (parseInt(text) / 100).toFixed(2);
}
```

## 6. Technical Implementation

### File Structure
```
index.html      # Single page with embedded React
app.js          # tool
style.css       # Basic styling (optional)
sample.txt      # Example ledger data
```


## 7. Limitations & Scope

### What It Does
✅ Parse basic ledger entries  
✅ Generate valid TEI-XML  
✅ Add minimal bookkeeping annotations  
✅ Allow manual data entry  
✅ Export XML files

## 8. Example Workflow

1. **Input**: Paste ledger text
```
October 3d 1828
John Deane D
To cash 6/    1 00
```

2. **Parse**: Extract fields
```json
{
  "date": "1828-10-03",
  "person": "John Deane",
  "type": "D",
  "description": "To cash 6/",
  "amount": "1.00"
}
```

3. **Generate**: Create TEI
```xml
<div type="entry">
  <date when="1828-10-03">October 3d 1828</date>
  <persName ana="#bk:Party">John Deane</persName>
  <note type="account-type">D</note>
  <desc>To cash 6/</desc>
  <measure ana="#bk:Money" quantity="1.00" unit="dollar">1 00</measure>
</div>
```

4. **Export**: Download as `ledger-entry.xml`

## Summary

This is a **minimal viable demo** that:
- Proves TEI can encode ledgers
- Shows bookkeeping ontology application
- Provides practical data entry tool
- Stays achievable in scope

Total complexity: ~300 lines of code, no backend required.