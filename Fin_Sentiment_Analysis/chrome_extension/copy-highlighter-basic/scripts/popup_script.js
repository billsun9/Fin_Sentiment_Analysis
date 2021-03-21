const updatePopup = () => {
    chrome.storage.sync.get(['current_data'], (data) => {
        if(data.current_data && data.current_data.length > 0) {
            data.current_data.forEach(cur_data => {
                const highlight = document.createElement('div');
                highlight.classList.add('highlight');
                highlight.innerHTML = cur_data;
                document.querySelector('#highlights').appendChild(highlight);
            })
        }
    });
}   
document.addEventListener('DOMContentLoaded', updatePopup);

document.querySelector('button').addEventListener('click', () => {
    chrome.runtime.sendMessage({
        message: 'send_all_data'
    }, response => {
        if (response.message === 'data_sent') {
            document.querySelector('#highlights').innerHTML = 'Your highlights here...';
        }
    });
})