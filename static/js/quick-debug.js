// Fast debug utility for dashboard
(function() {
    'use strict';
    
    const DEBUG = {
        errors: [],
        log: (msg, type = 'info') => {
            console.log(`[${type.toUpperCase()}] ${msg}`);
            if (type === 'error') DEBUG.errors.push({msg, time: Date.now()});
        },
        
        checkElements: () => {
            const critical = ['#tradingViewChart', '.connection-status', '#executeTrade'];
            critical.forEach(sel => {
                if (!document.querySelector(sel)) {
                    DEBUG.log(`Missing: ${sel}`, 'error');
                }
            });
        },
        
        panel: () => {
            const panel = document.createElement('div');
            panel.innerHTML = `
                <div style="position:fixed;top:10px;right:10px;background:rgba(0,0,0,0.9);color:white;padding:10px;border-radius:5px;z-index:10000;font-family:monospace;font-size:11px;">
                    <div>Errors: ${DEBUG.errors.length}</div>
                    <button onclick="DEBUG.checkElements()">Check DOM</button>
                    <button onclick="this.parentElement.remove()">×</button>
                </div>
            `;
            document.body.appendChild(panel);
        }
    };
    
    // Quick keyboard shortcut
    document.addEventListener('keydown', e => {
        if (e.ctrlKey && e.shiftKey && e.key === 'D') {
            e.preventDefault();
            DEBUG.panel();
        }
    });
    
    window.DEBUG = DEBUG;
})();
