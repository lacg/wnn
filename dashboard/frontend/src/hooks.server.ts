import type { HandleServerError } from '@sveltejs/kit';

// Handle unexpected errors gracefully
export const handleError: HandleServerError = async ({ error, event }) => {
  console.error('Server error:', error);

  // Return a user-friendly error message
  return {
    message: 'An unexpected error occurred. Please reload the page.',
  };
};
