# TEI Modeling for Laban Morey Wheaton Day Book (1828-1829 & 1831-1832)

## Document Structure

### Root Elements
- `<TEI>` with standard namespace declarations
- `<teiHeader>` for manuscript description and encoding decisions
- `<text><body>` containing chronological entries

### Organization
- `<pb n="[number]"/>` for page breaks
- `<div type="entry" when="[ISO-date]">` for each dated section
- `<div type="transaction" ana="[bookkeeping-ontology-ref]">` for individual transactions

## Transaction Encoding

### Basic Structure
```xml
<div type="transaction" ana="#debit|#credit" xml:id="[unique-id]">
  <persName ref="#[person-id]" n="[account-number]">[Name]</persName>
  <desc>[goods/services description]</desc>
  <measure type="currency" unit="[dollar|shilling]" quantity="[decimal]">
    [original notation]
  </measure>
  <note type="settled">Settled</note> <!-- when applicable -->
</div>
```

### Bookkeeping Ontology References (@ana)
- `#debit` - amounts owed to account holder
- `#credit` - amounts owed by account holder
- `#settlement` - settled/closed accounts
- `#order` - payment orders on third parties

## Core Components

### Personal Names
- `<persName ref="#[person-id]">` with unique identifier
- `@n` attribute for account numbers: `n="342"`
- Link name variants through `@ref` to single ID

### Financial Amounts
- `<measure type="currency" unit="dollar" quantity="[decimal]">`
- `@unit` values: `dollar`, `shilling`, `pence`
- `@quantity` as normalized decimal value
- Preserve original notation as element content

### Dates
- `<date when="1828-09-15">Monday September-15th-1828</date>`
- ISO 8601 format in `@when`
- Original formatting preserved in content

### Goods and Services
- Simple `<desc>` element for all descriptions
- Include quantities inline: `<desc>1 Buss. Rye</desc>`
- No need for complex type attributes

## Special Transaction Types

### Orders on Third Parties
```xml
<desc>order upon <persName ref="#smith">T. Smith</persName> for 3/</desc>
```

### Labor/Employment
```xml
<desc>work from this day at $13.00 pr month</desc>
```
Or for specific periods:
```xml
<desc>work from Oct. 14th 1828 16 Days</desc>
```

### Delivered Goods
```xml
<desc>ax dlvd By <persName ref="#puffer">Puffer</persName></desc>
```

## Settlement Notation
```xml
<note type="settled">Settled</note>
```
Place after the main transaction elements when accounts are marked as settled.

## Editorial Handling

### Name Variants
```xml
<persName ref="#wilbur">
  <choice>
    <orig>Wilber</orig>
    <reg>Wilbur</reg>
  </choice>
</persName>
```

### Uncertain Readings
```xml
<unclear>[text]</unclear>
```

### Common Abbreviations
Expand silently in most cases:
- "Buss." → "Bushel"
- "D" → recorded via `@ana="#debit"`
- "Cr" → recorded via `@ana="#credit"`

## Linking Related Transactions

### Basic Cross-References
- `@xml:id` on each transaction
- `@corresp` to link related debits/credits

### Example
```xml
<div type="transaction" ana="#debit" xml:id="t001" corresp="#t002">
  <persName ref="#drake">Derius Drake</persName>
  <desc>To ax dlvd By Puffer</desc>
  <measure type="currency" unit="dollar" quantity="1.50">1 50</measure>
</div>

<div type="transaction" ana="#credit" xml:id="t002" corresp="#t001">
  <persName ref="#puffer" n="109">Pliny Puffer</persName>
  <desc>By 1 ax dlv.d. Drake</desc>
  <measure type="currency" unit="dollar" quantity="1.50">1 50</measure>
</div>
```

## Authority Control

### Minimal Prosopography
Create a simple person list in the header or as a separate file:
```xml
<person xml:id="drake">
  <persName>Derius Drake</persName>
</person>
<person xml:id="wilbur">
  <persName>Amos Wilbur</persName>
  <note>Also appears as: Wilber, Wilham</note>
</person>
```

## What This Model Omits (Intentionally)

- Complex role attributes (debit/credit already indicates this)
- Elaborate transaction subtypes (the description tells us what it is)
- Nested measurement structures (inline descriptions work fine)
- Multiple classification systems (one @ana reference is enough)
- Special structures for rare cases (handle with `<note>` if needed)

## Implementation Notes

1. **Start simple**: Encode the basic transaction structure first
2. **Test with real data**: Try encoding 10-20 entries before adding complexity
3. **Add only what you need**: If you find yourself never using an attribute, remove it
4. **Document decisions**: Note in the header why you made specific choices
5. **Prioritize consistency**: Better to be consistently simple than inconsistently complex

## Complete Example Entry

```xml
<div type="entry" when="1828-09-15">
  <date when="1828-09-15">Monday September-15th-1828</date>
  
  <div type="transaction" ana="#debit" xml:id="t001">
    <note type="settled">Settled</note>
    <persName ref="#drake">Derius Drake</persName>
    <desc>To ax dlvd By Puffer 9/1</desc>
    <measure type="currency" unit="dollar" quantity="1.50">1 50</measure>
  </div>
  
  <div type="transaction" ana="#credit" xml:id="t002">
    <note type="settled">Settled</note>
    <persName ref="#drake">Derius Drake</persName>
    <desc>By cutting two sticks</desc>
  </div>
  
  <div type="transaction" ana="#credit" xml:id="t003">
    <persName ref="#puffer" n="109">Pliny Puffer</persName>
    <desc>By 1 ax dlv.d. Drake 9/</desc>
    <measure type="currency" unit="dollar" quantity="1.50">1 50</measure>
  </div>
</div>
```