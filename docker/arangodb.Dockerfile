FROM arangodb:3.11

# Install curl for health checks
USER root
RUN apk --no-cache add curl

