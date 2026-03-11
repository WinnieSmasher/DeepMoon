const fs = require('fs');
const file = 'C:/Users/sunshine/AppData/Roaming/npm/node_modules/@anthropic-ai/claude-code/cli.js';
let content = fs.readFileSync(file, 'utf8');
const original = content;

// Replace various possible forms of the validation
content = content.replace(/startsWith\("sk-ant-api03-"\)/g, 'startsWith("sk-")');
content = content.replace(/startsWith\('sk-ant-api03-'\)/g, "startsWith('sk-')");
content = content.replace(/startsWith\("sk-ant-"\)/g, 'startsWith("sk-")');
content = content.replace(/startsWith\('sk-ant-'\)/g, "startsWith('sk-')");

if (content !== original) {
    fs.writeFileSync(file, content, 'utf8');
    console.log('Patched API key validation in cli.js!');
} else {
    console.log('Could not find validation logic to patch. Looking for other patterns...');
    // Let's print out what startsWith we actually have
    const matches = content.match(/startsWith\([^)]+\)/g);
    if (matches) {
        console.log("Found these startsWith calls:");
        console.log([...new Set(matches)].join('\n'));
    }
}
