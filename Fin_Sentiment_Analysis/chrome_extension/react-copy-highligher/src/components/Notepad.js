import { useState } from 'react';
// import { useHistory } from 'react-router-dom';

const Notepad = () => {
    const [title, setTitle] = useState("");
    const [text, setText] = useState("");

    // const hist = useHistory();
    const sendMessage = () => {
        let request = new XMLHttpRequest();
        request.open("POST", "https://discord.com/api/webhooks/822156124916678676/3EmfzC8BtjHYrwScmB1way_wNNeqa37bM7rwbjODiuraLq9VSdHBM6bvJMiav0Dl_Fos");
    
        request.setRequestHeader('Content-type', 'application/json');
    
        const params = {
          username: "WZDM Chrome Extension - Notepad",
          avatar_url: "",
          content: JSON.stringify({title, text})
        }
    
        request.send(JSON.stringify(params));
    }

    const handleSubmit = (e) => {
        e.preventDefault();
        sendMessage();
        setTitle("");
        setText("");
    }
    return ( 
        <div className="notepad">
            <h2>Enter some text!</h2>
            <form onSubmit={handleSubmit}>
                <label>Title:</label>
                <input type="text" 
                    required 
                    value={title} 
                    onChange={(e) => setTitle(e.target.value)}
                />

                <label>Text body:</label>
                <textarea 
                    required 
                    value={text} 
                    onChange={(e) => setText(e.target.value)}
                ></textarea>
                <button>Add Data</button>
            </form>
        </div>
     );
}
 
export default Notepad;