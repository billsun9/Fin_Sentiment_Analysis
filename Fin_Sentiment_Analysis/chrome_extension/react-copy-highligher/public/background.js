let current_data = {};
current_data['current_data'] = [];

const sendMessage = () => {
    var request = new XMLHttpRequest();
    request.open("POST", "https://discord.com/api/webhooks/822156124916678676/3EmfzC8BtjHYrwScmB1way_wNNeqa37bM7rwbjODiuraLq9VSdHBM6bvJMiav0Dl_Fos");

    request.setRequestHeader('Content-type', 'application/json');

    var params = {
      username: "WZDM Chrome Extension - Jots",
      avatar_url: "",
      content: JSON.stringify(current_data)
    }

    request.send(JSON.stringify(params));
  }
  
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (/^http/.test(tab.url) && changeInfo.status === 'complete') {
        chrome.tabs.executeScript(tabId, { file: './foreground.js' }, () => {
            console.log('The foreground script has been injected.');
        });
    }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.message === 'add_copied_data') {
        current_data['current_data'].push(request.payload);
        chrome.storage.sync.set({ 'current_data': current_data });

        chrome.runtime.sendMessage({
            message: 'new_jot',
            payload: current_data['current_data']
        });

    } else if (request.message === 'send_data_to_server') {
        sendMessage();
        current_data['current_data'] = [];
        chrome.storage.sync.set({ 'current_data': current_data });
        sendResponse({ message: 'data_sent_to_server' })

    } else if (request.message === 'get_current_data_on_load') {
        sendResponse({ message: 'loaded_current_data', payload: current_data['current_data']})
    }
});