$(document).ready(function(){
    const socket = io();

    socket.on('connect', () => {
        socket.on('data', msg => {
            $('#central').append(`
                <div class="row epoch">
                    <span><b>Epoch</b> ${msg.num} of ${msg.total}</span>
                    <span><b>Loss</b> ${msg.loss}</span>
                </div>
            `)
        });
    });
});
