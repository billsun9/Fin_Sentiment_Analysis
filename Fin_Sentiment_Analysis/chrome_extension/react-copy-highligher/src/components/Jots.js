/*global chrome*/

import { useEffect, useState } from 'react';

const Jots = () => {
    const [allJots, setAllJots] = useState([]);
    
    const getId = () => {
        return Math.floor(Math.random() * 999999);
    }
    
    const handleSubmit = (e) => {
        e.preventDefault();
        if(allJots.length > 0) {
            chrome.runtime.sendMessage({ message: 'send_data_to_server' }, res => {
                if(res.message === 'data_sent_to_server') {
                    setAllJots([]);
                }
            });
        }
    }
    useEffect(() => {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if(request.message === 'new_jot') {
                setAllJots(request.payload);
            }
        });

        chrome.runtime.sendMessage({message: 'get_current_data_on_load'}, res => {
            if(res.message === 'loaded_current_data') {
                setAllJots(res.payload);
            }
        });
        // chrome.storage.sync.get(null, res => {
        //     let keys = Object.keys(res);
        //     if (keys.length != 0) {
        //         setAllJots(res['current_data']);
        //     }
        // });


    }, []);
    return ( 
        <div className="jots-container">
            <h2>Copy-pastes and Jots here!</h2>
            {allJots.length != 0 && allJots.map((jot) => (
                <div className="jot" key={getId}>
                    {jot}
                </div>
            ))}
            <form onSubmit={handleSubmit}>
                <button >Submit</button>
            </form>
            
        </div>
     );
}
 
export default Jots;