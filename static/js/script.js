document.addEventListener('DOMContentLoaded', function() {
            const toggleButton = document.getElementById('toggle-sidebar');
            const sidebar = document.getElementById('sidebar');
            toggleButton.addEventListener('click', function() {
                sidebar.classList.toggle('collapsed');
            });

            const clearButton = document.getElementById('clear-map');
            clearButton.addEventListener('click', function() {
                image_layers = [];
                shapefile_layers = [];
                location.reload();
            });
        });

 document.addEventListener('DOMContentLoaded', function() {
            const alertElement = document.getElementById('alert');
            if (alertElement) {
                setTimeout(function() {
                    alertElement.classList.remove('show');
                    alertElement.classList.add('hide');
                }, 5000);
            }
        });
