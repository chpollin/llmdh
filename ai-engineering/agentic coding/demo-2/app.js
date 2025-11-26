const { useState, useEffect } = React;

// Parser Functions
const monthMap = {
    'January': '01', 'February': '02', 'March': '03', 'April': '04',
    'May': '05', 'June': '06', 'July': '07', 'August': '08',
    'September': '09', 'Sept': '09', 'October': '10', 'Oct': '10',
    'November': '11', 'Nov': '11', 'December': '12', 'Dec': '12'
};

function parseDate(dateStr) {
    const patterns = [
        /(\w+)\s+(\w+)-(\d+)\w*-(\d{4})/,
        /(\w+)\s+(\w+)\.\s*(\d+)\w*\s+(\d{4})/,
        /(\w+)\s+(\d+)\s*-?\s*(\d{4})/,
        /(\w+)\s+(\d+)\w*\s+(\d{4})/
    ];
    
    for (const pattern of patterns) {
        const match = dateStr.match(pattern);
        if (match) {
            if (match.length === 5) {
                const month = monthMap[match[2]] || '01';
                const day = match[3].padStart(2, '0');
                return {
                    iso: `${match[4]}-${month}-${day}`,
                    text: dateStr
                };
            } else if (match.length === 4) {
                const month = monthMap[match[1]] || '01';
                const day = match[2].padStart(2, '0');
                return {
                    iso: `${match[3]}-${month}-${day}`,
                    text: dateStr
                };
            }
        }
    }
    return { iso: '', text: dateStr };
}

function parseAmount(amountStr) {
    if (!amountStr) return { decimal: 0, text: '' };
    
    amountStr = amountStr.trim();
    
    if (amountStr.includes('/')) {
        const shillings = parseInt(amountStr) || 0;
        return {
            decimal: (shillings / 6).toFixed(2),
            text: amountStr
        };
    }
    
    const spacedMatch = amountStr.match(/(\d+)\s+(\d{2})/);
    if (spacedMatch) {
        return {
            decimal: `${spacedMatch[1]}.${spacedMatch[2]}`,
            text: amountStr
        };
    }
    
    const decimalMatch = amountStr.match(/(\d+)\.(\d{2})/);
    if (decimalMatch) {
        return {
            decimal: amountStr,
            text: amountStr
        };
    }
    
    const centsMatch = amountStr.match(/^(\d{2})$/);
    if (centsMatch) {
        return {
            decimal: `0.${centsMatch[1]}`,
            text: amountStr
        };
    }
    
    return {
        decimal: (parseFloat(amountStr) || 0).toFixed(2),
        text: amountStr
    };
}

function parseEntry(text) {
    const lines = text.split('\n').filter(line => line.trim());
    const entry = {
        date: '',
        dateText: '',
        person: '',
        accountType: '',
        description: '',
        amount: 0,
        amountText: ''
    };
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        const datePatterns = [
            /^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)/i,
            /^(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Oct|Nov|Dec)/i
        ];
        
        if (datePatterns.some(pattern => pattern.test(line))) {
            const parsed = parseDate(line);
            entry.date = parsed.iso;
            entry.dateText = parsed.text;
            continue;
        }
        
        const personPattern = /^(?:Settled\s+)?(?:\d+\s+)?([A-Z][a-zA-Z\s]+?)\s+(DR|D|CR|C|Cr|Settled)/;
        const match = line.match(personPattern);
        if (match) {
            entry.person = match[1].trim();
            entry.accountType = match[2].toUpperCase().replace('CR', 'Cr');
            continue;
        }
        
        const descPattern = /^(?:To|By)\s+(.*?)(?:\s+(\d+\s+\d{2}|\d+\.\d{2}|\d+\/|\d{2}))?$/;
        const descMatch = line.match(descPattern);
        if (descMatch) {
            entry.description = descMatch[1].trim();
            if (descMatch[2]) {
                const parsed = parseAmount(descMatch[2]);
                entry.amount = parseFloat(parsed.decimal);
                entry.amountText = parsed.text;
            }
        }
    }
    
    return entry;
}

function entryToTEI(entry) {
    let tei = `  <div type="entry">\n`;
    
    if (entry.dateText) {
        tei += `    <date when="${entry.date}">${entry.dateText}</date>\n`;
    }
    
    if (entry.person) {
        tei += `    <persName ana="#bk:Party">${entry.person}</persName>\n`;
    }
    
    if (entry.accountType) {
        tei += `    <note type="account-type">${entry.accountType}</note>\n`;
    }
    
    if (entry.description) {
        tei += `    <desc>${entry.description}</desc>\n`;
    }
    
    if (entry.amount > 0) {
        tei += `    <measure ana="#bk:Money" quantity="${entry.amount}" unit="dollar">${entry.amountText}</measure>\n`;
    }
    
    tei += `  </div>`;
    
    return tei;
}

function generateFullTEI(entries) {
    // Extract unique persons for the particDesc
    const uniquePersons = [...new Set(entries.map(e => e.person).filter(p => p))];
    const personList = uniquePersons.map((person, index) => 
        `          <person xml:id="person_${index + 1}" role="customer">
            <persName>${person}</persName>
            <note>Account holder in 1828 ledger</note>
          </person>`
    ).join('\n');
    
    // Calculate totals
    const totalDebits = entries
        .filter(e => ['DR', 'D'].includes(e.accountType))
        .reduce((sum, e) => sum + (e.amount || 0), 0);
    const totalCredits = entries
        .filter(e => ['CR', 'C', 'Cr'].includes(e.accountType))
        .reduce((sum, e) => sum + (e.amount || 0), 0);
    
    // Get date range
    const validDates = entries.filter(e => e.date).map(e => e.date).sort();
    const dateFrom = validDates[0] || '1828-01-01';
    const dateTo = validDates[validDates.length - 1] || '1828-12-31';
    
    let tei = `<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" 
     xmlns:bk="http://www.tei-c.org/ns/bookkeeping/1.0"
     xml:lang="en">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title type="main">Account Ledger - 1828</title>
        <title type="sub">Digital Edition of Historical Bookkeeping Records</title>
        <author>
          <persName>Unknown Merchant</persName>
          <note>Original ledger keeper, likely a general store owner</note>
        </author>
        <respStmt>
          <resp>TEI encoding and digitization by</resp>
          <name>Historical Ledger Entry System</name>
          <note>Automated transcription tool with manual verification</note>
        </respStmt>
      </titleStmt>
      <publicationStmt>
        <publisher>Digital Humanities Archive</publisher>
        <pubPlace>Online</pubPlace>
        <date when="${new Date().toISOString().split('T')[0]}">Encoded on ${new Date().toLocaleDateString()}</date>
        <availability status="free">
          <licence target="https://creativecommons.org/licenses/by/4.0/">
            Creative Commons Attribution 4.0 International
          </licence>
        </availability>
      </publicationStmt>
      <sourceDesc>
        <msDesc>
          <msIdentifier>
            <repository>Local Historical Society</repository>
            <collection>19th Century Business Records</collection>
            <idno>MS-1828-001</idno>
          </msIdentifier>
          <msContents>
            <summary>
              Account ledger containing ${entries.length} transactions from ${dateFrom} to ${dateTo}.
              Total debits: $${totalDebits.toFixed(2)}, Total credits: $${totalCredits.toFixed(2)}
            </summary>
            <msItem>
              <title>Daily Transaction Records</title>
              <note>Entries include goods sold, services rendered, and cash transactions</note>
            </msItem>
          </msContents>
          <physDesc>
            <objectDesc form="codex">
              <supportDesc material="paper">
                <support>Hand-made paper with visible chain lines</support>
                <extent>${entries.length} entries digitized from original manuscript</extent>
                <condition>Fair to good, some ink fading and paper foxing</condition>
              </supportDesc>
            </objectDesc>
            <handDesc>
              <handNote xml:id="h1" script="cursive" medium="iron-gall-ink">
                Primary hand, likely the store owner
              </handNote>
            </handDesc>
          </physDesc>
          <history>
            <origin>
              <origDate from="${dateFrom}" to="${dateTo}">1828</origDate>
              <origPlace>New England, United States</origPlace>
            </origin>
          </history>
        </msDesc>
      </sourceDesc>
    </fileDesc>
    <encodingDesc>
      <projectDesc>
        <p>This TEI edition is part of a digital humanities project to preserve and analyze
           19th century American business records, providing insights into local economies,
           trade patterns, and social networks of the early Republic period.</p>
      </projectDesc>
      <classDecl>
        <taxonomy xml:id="account-types">
          <category xml:id="debit">
            <catDesc>Debit entries (DR/D) - amounts owed to the ledger keeper</catDesc>
          </category>
          <category xml:id="credit">
            <catDesc>Credit entries (CR/C) - amounts owed by the ledger keeper</catDesc>
          </category>
          <category xml:id="settled">
            <catDesc>Settled accounts - balanced or closed transactions</catDesc>
          </category>
        </taxonomy>
        <taxonomy xml:id="commodities">
          <category xml:id="goods">
            <catDesc>Physical goods (tools, food, materials)</catDesc>
          </category>
          <category xml:id="services">
            <catDesc>Labor and services</catDesc>
          </category>
          <category xml:id="cash">
            <catDesc>Monetary transactions</catDesc>
          </category>
        </taxonomy>
      </classDecl>
      <tagsDecl>
        <namespace name="http://www.tei-c.org/ns/bookkeeping/1.0">
          <tagUsage gi="bk:Party">Marks persons or entities involved in transactions</tagUsage>
          <tagUsage gi="bk:Money">Marks monetary amounts with normalized values</tagUsage>
          <tagUsage gi="bk:Commodity">Marks goods and services exchanged</tagUsage>
        </namespace>
      </tagsDecl>
    </encodingDesc>
    <profileDesc>
      <langUsage>
        <language ident="en-US">American English, early 19th century</language>
      </langUsage>
      <particDesc>
        <listPerson>
${personList}
        </listPerson>
      </particDesc>
      <settingDesc>
        <setting>
          <name>General Store or Trading Post</name>
          <date from="1828-01-01" to="1828-12-31">1828</date>
          <locale>Rural New England community</locale>
        </setting>
      </settingDesc>
    </profileDesc>
    <revisionDesc>
      <change when="${new Date().toISOString()}" who="#system">
        Initial TEI encoding from transcribed ledger entries
      </change>
    </revisionDesc>
  </teiHeader>
  <text>
    <front>
      <div type="summary">
        <head>Ledger Summary</head>
        <list>
          <item>Total Entries: ${entries.length}</item>
          <item>Date Range: ${dateFrom} to ${dateTo}</item>
          <item>Unique Account Holders: ${uniquePersons.length}</item>
          <item>Total Debits: $${totalDebits.toFixed(2)}</item>
          <item>Total Credits: $${totalCredits.toFixed(2)}</item>
          <item>Balance: $${(totalDebits - totalCredits).toFixed(2)}</item>
        </list>
      </div>
    </front>
    <body>
      <div type="ledger">
        <head>Transaction Entries</head>
`;
    
    entries.forEach((entry, index) => {
        tei += entryToEnhancedTEI(entry, index + 1, uniquePersons) + '\n';
    });
    
    tei += `      </div>
    </body>
    <back>
      <div type="index">
        <head>Index of Persons</head>
        <list>
${uniquePersons.map(person => `          <item>${person}</item>`).join('\n')}
        </list>
      </div>
    </back>
  </text>
</TEI>`;
    
    return tei;
}

function entryToEnhancedTEI(entry, entryNum, allPersons) {
    const personId = allPersons.indexOf(entry.person) + 1;
    let tei = `        <div type="entry" n="${entryNum}" xml:id="entry_${entryNum}">`;
    
    if (entry.dateText) {
        if (entry.date) {
            tei += `
          <date when="${entry.date}" cert="high">${entry.dateText}</date>`;
        } else {
            tei += `
          <date cert="low">${entry.dateText}</date>`;
        }
    }
    
    if (entry.person) {
        tei += `
          <persName ref="#person_${personId}" ana="#bk:Party" role="account-holder">
            ${entry.person}
          </persName>`;
    }
    
    if (entry.accountType) {
        const typeCategory = ['DR', 'D'].includes(entry.accountType) ? 'debit' : 
                           ['CR', 'C', 'Cr'].includes(entry.accountType) ? 'credit' : 'settled';
        tei += `
          <note type="account-type" subtype="${typeCategory}">
            <term>${entry.accountType}</term>
          </note>`;
    }
    
    if (entry.description) {
        // Analyze description for commodity types
        const commodityType = entry.description.toLowerCase().includes('cash') ? 'cash' :
                            entry.description.toLowerCase().includes('work') || 
                            entry.description.toLowerCase().includes('drawing') ? 'services' : 'goods';
        
        tei += `
          <seg type="transaction" ana="#${commodityType}">
            <desc>${entry.description}</desc>`;
        
        // Extract potential commodity mentions
        if (entry.description.match(/\b(ax|axe|logs?|nails?|rye|potatoes?|buss\.?)\b/i)) {
            const commodity = entry.description.match(/\b(ax|axe|logs?|nails?|rye|potatoes?|buss\.?)\b/i)[0];
            tei += `
            <name type="commodity" ana="#bk:Commodity">${commodity}</name>`;
        }
        
        tei += `
          </seg>`;
    }
    
    if (entry.amount > 0) {
        tei += `
          <measure type="currency" ana="#bk:Money" quantity="${entry.amount}" unit="USD" commodity="currency">
            <num>${entry.amountText}</num>
            <note>$${entry.amount} in decimal notation</note>
          </measure>`;
    }
    
    tei += `
        </div>`;
    
    return tei;
}

// React Components
function LedgerEntryForm({ onAddEntry }) {
    const [formData, setFormData] = useState({
        dateText: '',
        person: '',
        accountType: 'DR',
        description: '',
        amountText: ''
    });
    
    const handleSubmit = (e) => {
        e.preventDefault();
        const parsed = parseDate(formData.dateText);
        const amount = parseAmount(formData.amountText);
        
        const entry = {
            date: parsed.iso,
            dateText: formData.dateText,
            person: formData.person,
            accountType: formData.accountType,
            description: formData.description,
            amount: parseFloat(amount.decimal),
            amountText: amount.text
        };
        
        onAddEntry(entry);
        setFormData({
            dateText: '',
            person: '',
            accountType: 'DR',
            description: '',
            amountText: ''
        });
    };
    
    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };
    
    return (
        <form className="entry-form" onSubmit={handleSubmit}>
            <h3>Manual Entry</h3>
            <div className="form-group">
                <label>Date (e.g., Monday September-15th-1828)</label>
                <input
                    type="text"
                    name="dateText"
                    value={formData.dateText}
                    onChange={handleChange}
                    placeholder="Monday September-15th-1828"
                    required
                />
            </div>
            <div className="form-group">
                <label>Person/Entity</label>
                <input
                    type="text"
                    name="person"
                    value={formData.person}
                    onChange={handleChange}
                    placeholder="Derius Drake"
                    required
                />
            </div>
            <div className="form-group">
                <label>Account Type</label>
                <select
                    name="accountType"
                    value={formData.accountType}
                    onChange={handleChange}
                >
                    <option value="DR">DR (Debit)</option>
                    <option value="D">D (Debit)</option>
                    <option value="Cr">Cr (Credit)</option>
                    <option value="C">C (Credit)</option>
                    <option value="Settled">Settled</option>
                </select>
            </div>
            <div className="form-group">
                <label>Description</label>
                <input
                    type="text"
                    name="description"
                    value={formData.description}
                    onChange={handleChange}
                    placeholder="To ax dlvd By Puffer 9/1"
                />
            </div>
            <div className="form-group">
                <label>Amount (e.g., 1 50 or 6/)</label>
                <input
                    type="text"
                    name="amountText"
                    value={formData.amountText}
                    onChange={handleChange}
                    placeholder="1 50"
                />
            </div>
            <button type="submit">Add Entry</button>
        </form>
    );
}

function TextParser({ onAddEntry }) {
    const [text, setText] = useState('');
    
    const handleParse = () => {
        const entry = parseEntry(text);
        if (entry.person) {
            onAddEntry(entry);
            setText('');
        }
    };
    
    return (
        <div className="text-parser">
            <h3>Parse Text Entry</h3>
            <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste ledger entry text here...
Monday September-15th-1828
Derius Drake DR
To ax dlvd By Puffer 9/1    1 50"
                rows="6"
            />
            <button onClick={handleParse}>Parse & Add</button>
        </div>
    );
}

function EntryList({ entries, onDeleteEntry }) {
    return (
        <div className="entry-list">
            <h3>Entries ({entries.length})</h3>
            {entries.length === 0 ? (
                <p className="no-entries">No entries yet. Add some using the form above.</p>
            ) : (
                <div className="entries">
                    {entries.map((entry, index) => (
                        <div key={index} className="entry-item">
                            <div className="entry-header">
                                <span className="entry-date">{entry.dateText || 'No date'}</span>
                                <button 
                                    className="delete-btn"
                                    onClick={() => onDeleteEntry(index)}
                                >
                                    ×
                                </button>
                            </div>
                            <div className="entry-details">
                                <span className="entry-person">{entry.person}</span>
                                <span className="entry-type">{entry.accountType}</span>
                                {entry.amount > 0 && (
                                    <span className="entry-amount">${entry.amount}</span>
                                )}
                            </div>
                            {entry.description && (
                                <div className="entry-desc">{entry.description}</div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function App() {
    const [entries, setEntries] = useState([]);
    const [showPreview, setShowPreview] = useState(false);
    
    const handleAddEntry = (entry) => {
        setEntries([...entries, entry]);
    };
    
    const handleDeleteEntry = (index) => {
        setEntries(entries.filter((_, i) => i !== index));
    };
    
    const handleExport = () => {
        const tei = generateFullTEI(entries);
        const blob = new Blob([tei], { type: 'text/xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ledger-${new Date().toISOString().split('T')[0]}.xml`;
        a.click();
        URL.revokeObjectURL(url);
    };
    
    const handleLoadSample = () => {
        const sampleEntries = [
            {
                date: "1828-09-15",
                dateText: "Monday September-15th-1828",
                person: "Derius Drake",
                accountType: "DR",
                description: "To ax dlvd By Puffer 9/1",
                amount: 1.50,
                amountText: "1 50"
            },
            {
                date: "1828-10-03",
                dateText: "October 3d 1828",
                person: "John Deane",
                accountType: "D",
                description: "To cash 6/",
                amount: 1.00,
                amountText: "1 00"
            },
            {
                date: "1828-10-03",
                dateText: "October 3d 1828",
                person: "Thomas Danforth 2d",
                accountType: "D",
                description: "To 1 lb. 4d Cut Nails .8 1 Buss. Rye 90",
                amount: 0.98,
                amountText: "98"
            }
        ];
        setEntries(sampleEntries);
    };
    
    return (
        <div className="app">
            <header>
                <h1>Historical Ledger Entry System</h1>
                <p>1828 Account Book Digitization Tool</p>
            </header>
            
            <div className="main-content">
                <div className="input-section">
                    <LedgerEntryForm onAddEntry={handleAddEntry} />
                    <TextParser onAddEntry={handleAddEntry} />
                </div>
                
                <div className="output-section">
                    <EntryList 
                        entries={entries} 
                        onDeleteEntry={handleDeleteEntry}
                    />
                    
                    <div className="actions">
                        <button onClick={handleLoadSample} className="btn-secondary">
                            Load Sample Data
                        </button>
                        <button 
                            onClick={() => setShowPreview(!showPreview)}
                            className="btn-secondary"
                        >
                            {showPreview ? 'Hide' : 'Show'} TEI Preview
                        </button>
                        <button 
                            onClick={handleExport}
                            disabled={entries.length === 0}
                            className="btn-primary"
                        >
                            Export TEI-XML
                        </button>
                    </div>
                    
                    {showPreview && entries.length > 0 && (
                        <div className="tei-preview">
                            <h3>TEI-XML Preview</h3>
                            <pre>{generateFullTEI(entries)}</pre>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root'));