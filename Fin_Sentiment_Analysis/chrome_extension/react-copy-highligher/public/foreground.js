document.addEventListener('copy', () => {
    navigator.clipboard.readText()
        .then(res => {
            chrome.runtime.sendMessage({
                message: 'add_copied_data',
                payload: `${res}`
            })
        })
        .catch(err => console.log(err));
});